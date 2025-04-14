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
from bot.utils.chat.models import AIModel
from bot.utils.chat.processor import clean_response_output, save_message
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


async def execute_bing_search(query: str, user_prompt: str) -> dict[str, Any]:
    tool_instance = BingSearchTool()
    try:
        return await tool_instance.run(query=query, user_prompt=user_prompt)
    except Exception as e:
        logger.exception("[Search] - Error executing Bing search")
        return {"error": f"Error during search: {e}", "user_prompt": user_prompt}


def format_sources(results: list[dict[str, str]]) -> str:
    if not results:
        return "No sources found."

    sources = []
    for i, result in enumerate(results[:5], 1):  # Limit to top 5 sources
        title = result.get("title", "Untitled")
        url = result.get("url", "#")
        sources.append(f"{i}. [{title}]({url})")

    return "\n\n## Sources\n" + "\n".join(sources)


def extract_context_from_results(results: list[dict[str, str]], max_results: int = 5) -> str:
    context_parts = []
    for result in results[:max_results]:
        content = result.get("content", "").strip()
        snippet = result.get("snippet", "").strip()
        if content:
            context_parts.append(content)
        elif snippet:
            context_parts.append(snippet)

    return "\n\n".join(context_parts)


def generate_search_prompt(query: str, context: str) -> str:
    return (
        f'I performed a web search for: "{query}"\n\n'
        f"Here are the relevant results:\n\n{context}\n\n"
        f"Based on these search results, please provide a comprehensive answer to my "
        f"original query. DO NOT include a sources section at the end - I will add this myself."
    )


async def generate_ai_response(
    system_message: SystemMessage, prompt: str, model: AIModel
) -> str | None:
    messages = [system_message, UserMessage(content=prompt)]

    try:
        response = await azure_client.complete(
            messages=messages,
            model=model.value,
            tools=[],
        )
        if response.choices:
            return response.choices[0].message.content
        return None
    except Exception:
        logger.exception("[Search] - Error generating AI response")
        return None


async def process_and_send_response(
    message: Message, reply_text: str, sources_section: str, model: AIModel, user_query: str
) -> None:
    clean_response = clean_response_output(reply_text)
    full_response = f"[🔍 {model.value}] {clean_response}\n{sources_section}"

    if message.from_user:
        await save_message(message.from_user.id, user_query, clean_response)

    chunks = split_text_with_formatting(telegram_format(full_response))
    for chunk in chunks:
        await message.answer(chunk)


async def search_and_respond(message: Message, clean_text: str) -> None:
    if not message.from_user:
        await message.answer("Error: User information not available.")
        return

    model = AIModel.GPT_4O

    search_results = await execute_bing_search(clean_text, clean_text)
    if "error" in search_results:
        await message.answer(f"Error: {search_results['error']}")
        return

    results = search_results.get("results", [])
    if not results:
        await message.answer("No search results found for your query.")
        return

    combined_context = extract_context_from_results(results)

    prompt = generate_search_prompt(clean_text, combined_context)

    system_message = get_system_message_without_tools(message.from_user)
    logger.debug("[Search] - System message content: %s", system_message.content)

    try:
        reply_text = await generate_ai_response(system_message, prompt, model)

    except HttpResponseError as error:
        if error.status_code == 429:
            logger.warning("[Search] - Error with GPT-4o, falling back to GPT-4o-mini")
            model = AIModel.GPT_4O_MINI
            reply_text = await generate_ai_response(system_message, prompt, model)

    if not reply_text:
        await message.answer("Error generating response. Please try again later.")
        return

    sources_section = format_sources(results)

    await process_and_send_response(message, reply_text, sources_section, model, clean_text)


@router.message(Command("search"))
async def search_handler(message: Message, command: CommandObject) -> None:
    if not message.from_user:
        return

    if command.args is None or not command.args.strip():
        await message.answer(
            "Please provide a search query. Example: `/search latest AI developments`"
        )
        return

    await search_and_respond(message, command.args)
