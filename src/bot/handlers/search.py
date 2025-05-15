# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

import logging
import re
from datetime import UTC, datetime
from typing import Any

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


def get_system_message_without_tools(user: User) -> SystemMessage:
    if not user or not user.full_name:
        return SystemMessage(content=get_base_message())

    base_message = get_base_message()
    pattern = r"# Tools\s+.*?namespace functions \{.*?\} // namespace functions"
    modified_message = re.sub(pattern, "", base_message, flags=re.DOTALL).strip()

    search_note = (
        "\n\n# Search\n\nYou will respond based only on the web search results provided. "
        "Do not use your training data."
    )
    modified_message += search_note

    current_utc = datetime.now(UTC).strftime("%d-%m-%Y %H:%M:%S")
    language_code = user.language_code or "Unknown"
    lang_info = get_user_locale_info(language_code)
    session_info = format_session_info(user, current_utc, lang_info)

    return SystemMessage(content=f"{modified_message}\n_session:\n{session_info}")


async def execute_search(query: str) -> dict[str, Any]:
    tool_instance = BingSearchTool()
    try:
        return await tool_instance.run(query=query, user_prompt=query)
    except Exception as e:
        logger.exception("[Search] - Error executing Bing search")
        return {"error": f"Error during search: {e}", "user_prompt": query}


def extract_context(results: list[dict[str, str]], max_results: int = 5) -> str:
    context_parts = []
    for result in results[:max_results]:
        content = result.get("content", "").strip()
        snippet = result.get("snippet", "").strip()
        if content:
            context_parts.append(content)
        elif snippet:
            context_parts.append(snippet)
    return "\n\n".join(context_parts)


def format_sources(results: list[dict[str, str]]) -> str:
    if not results:
        return "No sources found."

    seen = set()
    sources = []
    for idx, result in enumerate(results[:5], 1):
        url = result.get("url", "#")
        if url in seen:
            continue
        seen.add(url)
        title = result.get("title", "Untitled")
        sources.append(f"{idx}. [{title}]({url})")

    return "\n\n<b>Sources:</b>\n" + "\n".join(sources)


def generate_prompt(query: str, context: str) -> str:
    return (
        f'Web search for: "{query}".\n'
        f"Relevant results (summarize concisely):\n{context}\n\n"
        f"Answer the query using only these results. Be concise and clear. "
        f"Do not include a sources section."
    )


def select_best_model() -> AIModel:
    if not rate_limit_tracker.is_rate_limited(AIModel.GPT_4_1):
        return AIModel.GPT_4_1
    return AIModel.GPT_4_1_MINI


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
            logger.warning("[Search] - Rate limit hit for model %s", model.value)
            rate_limit_tracker.set_rate_limited(model, 300)
        raise
    except Exception as e:
        logger.exception("[Search] - Unexpected error with %s: %s", model.value, str(e))
        return None


async def send_response(
    message: Message, reply_text: str, sources_section: str, model: AIModel, user_query: str
) -> None:
    if not message.from_user:
        return

    clean_response = clean_response_output(reply_text)
    full_response = f"[🔍 {model.value}] {clean_response}\n\n{sources_section}"

    await save_message(message.from_user.id, user_query, clean_response)
    chunks = split_text_with_formatting(full_response)
    for chunk in chunks:
        await message.answer(telegram_format(chunk), parse_mode="HTML")


async def handle_search_request(message: Message, query: str) -> None:
    if not message.from_user:
        return

    search_results = await execute_search(query)
    if "error" in search_results:
        await message.answer(f"Search error: {search_results['error']}")
        return

    results = search_results.get("results", [])
    if not results:
        await message.answer("No results found for your query.")
        return

    context = extract_context(results)
    prompt = generate_prompt(query, context)
    system_message = get_system_message_without_tools(message.from_user)

    model = select_best_model()
    reply_text = None

    try:
        reply_text = await generate_response(system_message, prompt, model)
    except HttpResponseError:
        if model != AIModel.GPT_4_1_MINI:
            logger.info("[Search] - Falling back to GPT-4.1-mini")
            model = AIModel.GPT_4_1_MINI
            try:
                reply_text = await generate_response(system_message, prompt, model)
            except Exception as e:
                logger.exception("[Search] - Fallback also failed: %s", str(e))

    if not reply_text:
        await message.answer("Could not generate a response. Please try again later.")
        return

    sources_section = format_sources(results)
    await send_response(message, reply_text, sources_section, model, query)


@router.message(Command("search"))
async def search_handler(message: Message, command: CommandObject) -> None:
    if not message.from_user:
        return

    if not command.args or not command.args.strip():
        await message.answer("Send a search query. Example: /search latest AI developments")
        return

    await handle_search_request(message, command.args.strip())
