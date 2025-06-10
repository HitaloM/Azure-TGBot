# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from aiogram import Router
from aiogram.filters import Command, CommandObject
from aiogram.types import Message, User
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.exceptions import HttpResponseError
from chatgpt_md_converter import telegram_format

from bot.utils.chat.client import azure_client
from bot.utils.chat.client.rate_limiter import rate_limit_tracker
from bot.utils.chat.models import AIModel
from bot.utils.chat.response_processor import clean_response_output, save_message
from bot.utils.chat.system_message import format_session_info, get_base_message
from bot.utils.chat.tools.bing_search import BingSearchTool
from bot.utils.text_splitter import split_text_with_formatting
from bot.utils.user_info import get_user_locale_info

logger = logging.getLogger(__name__)

router = Router(name="search")


class SearchConfig:
    MAX_RESULTS = 5
    MAX_CONTEXT_LENGTH = 4000
    FALLBACK_TIMEOUT = 300
    EXAMPLE_QUERY = "latest AI developments"


class SearchError(Exception):
    pass


class SearchTimeoutError(SearchError):
    pass


class SearchRateLimitError(SearchError):
    pass


@dataclass
class SearchResult:
    results: list[dict[str, str]]
    query: str
    error: str | None = None

    @property
    def has_error(self) -> bool:
        return self.error is not None

    @property
    def has_results(self) -> bool:
        return bool(self.results) and not self.has_error


class SystemMessageBuilder:
    @staticmethod
    def build_search_system_message(user: User) -> SystemMessage:
        if not user or not user.full_name:
            return SystemMessage(content=SystemMessageBuilder._get_base_search_message())

        search_note = (
            "\n\n# Search\n\nYou will respond based only on the web search results provided. "
            "Do not use your training data."
        )
        modified_message = SystemMessageBuilder._get_base_search_message() + search_note

        current_utc = datetime.now(UTC).strftime("%d-%m-%Y %H:%M:%S")
        language_code = user.language_code or "Unknown"
        lang_info = get_user_locale_info(language_code)
        session_info = format_session_info(user, current_utc, lang_info)

        return SystemMessage(content=f"{modified_message}\n_session:\n{session_info}")

    @staticmethod
    def _get_base_search_message() -> str:
        return get_base_message()


class SearchService:
    def __init__(self):
        self.search_tool = BingSearchTool()

    async def execute_search(self, query: str) -> SearchResult:
        try:
            logger.info("Executing search for query: %s", query)
            result = await self.search_tool.run(query=query, user_prompt=query)

            if "error" in result:
                logger.error("Search returned error: %s", result["error"])
                return SearchResult(results=[], query=query, error=result["error"])

            results = result.get("results", [])
            logger.info("Search completed with %d results", len(results))
            return SearchResult(results=results, query=query)

        except Exception as e:
            error_msg = f"Error during search: {e}"
            logger.exception("Search execution failed for query: %s", query)
            return SearchResult(results=[], query=query, error=error_msg)

    @staticmethod
    def extract_context(results: list[dict[str, str]]) -> str:
        if not results:
            return ""

        context_parts = []
        total_length = 0

        for result in results[: SearchConfig.MAX_RESULTS]:
            content = result.get("content", "").strip()
            snippet = result.get("snippet", "").strip()

            text_to_add = content or snippet
            if text_to_add and total_length + len(text_to_add) <= SearchConfig.MAX_CONTEXT_LENGTH:
                context_parts.append(text_to_add)
                total_length += len(text_to_add)
            elif total_length >= SearchConfig.MAX_CONTEXT_LENGTH:
                break

        return "\n\n".join(context_parts)

    @staticmethod
    def generate_prompt(query: str, context: str) -> str:
        return (
            f'Web search for: "{query}".\n'
            f"Relevant results (summarize concisely):\n{context}\n\n"
            f"Answer the query using only these results. Be concise and clear. "
            f"Do not include a sources section."
        )


class ResponseFormatter:
    @staticmethod
    def format_sources(results: list[dict[str, str]]) -> str:
        if not results:
            return "No sources found."

        seen = set()
        sources = []
        for idx, result in enumerate(results[: SearchConfig.MAX_RESULTS], 1):
            url = result.get("url", "#")
            if url in seen:
                continue
            seen.add(url)
            title = result.get("title", "Untitled")
            sources.append(f"{idx}. [{title}]({url})")

        return "\n\n*Sources:*\n" + "\n".join(sources)

    @staticmethod
    def format_full_response(reply_text: str, sources_section: str, model: AIModel) -> str:
        clean_response = clean_response_output(reply_text)
        return f"[🔍 {model.value}] {clean_response}\n\n{sources_section}"


class ModelSelector:
    @staticmethod
    def select_best_model() -> AIModel:
        if not rate_limit_tracker.is_rate_limited(AIModel.GPT_4_1):
            return AIModel.GPT_4_1
        return AIModel.GPT_4_1_MINI


class AIResponseGenerator:
    @staticmethod
    async def generate_response(
        system_message: SystemMessage, prompt: str, model: AIModel
    ) -> str | None:
        messages = [system_message, UserMessage(content=prompt)]

        try:
            response = await azure_client.complete(
                messages=messages,
                model=model.value,
                tools=[],
            )
            if not response.choices:
                return None
            return response.choices[0].message.content
        except HttpResponseError as error:
            if error.status_code == 429:
                logger.warning("Rate limit hit for model %s", model.value)
                rate_limit_tracker.set_rate_limited(model, SearchConfig.FALLBACK_TIMEOUT)
            raise
        except Exception as e:
            logger.exception("Unexpected error with %s: %s", model.value, str(e))
            return None

    @staticmethod
    async def generate_with_fallback(
        system_message: SystemMessage, prompt: str
    ) -> tuple[str | None, AIModel]:
        model = ModelSelector.select_best_model()
        reply_text = None

        try:
            reply_text = await AIResponseGenerator.generate_response(system_message, prompt, model)
        except HttpResponseError:
            if model != AIModel.GPT_4_1_MINI:
                logger.info("Falling back to GPT-4.1-mini")
                model = AIModel.GPT_4_1_MINI
                try:
                    reply_text = await AIResponseGenerator.generate_response(
                        system_message, prompt, model
                    )
                except Exception as e:
                    logger.exception("Fallback also failed: %s", str(e))

        return reply_text, model


class MessageHandler:
    def __init__(self):
        self.search_service = SearchService()
        self.formatter = ResponseFormatter()

    async def send_response(
        self,
        message: Message,
        reply_text: str,
        sources_section: str,
        model: AIModel,
        user_query: str,
    ) -> None:
        if not message.from_user:
            return

        full_response = self.formatter.format_full_response(reply_text, sources_section, model)

        await save_message(
            message.from_user.id, message.chat.id, user_query, clean_response_output(reply_text)
        )
        chunks = split_text_with_formatting(full_response)
        for chunk in chunks:
            await message.reply(telegram_format(chunk), parse_mode="HTML")

    async def handle_search_request(self, message: Message, query: str) -> None:
        if not message.from_user:
            return

        search_result = await self.search_service.execute_search(query)

        if search_result.has_error:
            await message.reply(f"Search error: {search_result.error}")
            return

        if not search_result.has_results:
            await message.reply("No results found for your query.")
            return

        context = self.search_service.extract_context(search_result.results)
        prompt = self.search_service.generate_prompt(query, context)
        system_message = SystemMessageBuilder.build_search_system_message(message.from_user)

        reply_text, model = await AIResponseGenerator.generate_with_fallback(
            system_message, prompt
        )

        if not reply_text:
            await message.reply("Could not generate a response. Please try again later.")
            return

        sources_section = self.formatter.format_sources(search_result.results)
        await self.send_response(message, reply_text, sources_section, model, query)


message_handler = MessageHandler()


@router.message(Command("search"))
async def search_handler(message: Message, command: CommandObject) -> None:
    if not message.from_user:
        return

    if not command.args or not command.args.strip():
        example_text = f"Send a search query. Example: /search {SearchConfig.EXAMPLE_QUERY}"
        await message.reply(example_text)
        return

    await message_handler.handle_search_request(message, command.args.strip())
