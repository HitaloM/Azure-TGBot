# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

import re
import tempfile
from difflib import get_close_matches
from pathlib import Path
from typing import Any

from aiogram.types import File, Message
from azure.ai.inference.models import UserMessage
from azure.core.exceptions import HttpResponseError
from chatgpt_md_converter import telegram_format

from bot.database import prune_conversation_history, save_conversation
from bot.database.operations import clear_user_conversation_history
from bot.utils.text_splitter import split_text_with_formatting

from .client import DEFAULT_MODEL, query_azure_chat, query_azure_chat_with_image
from .context import build_reply_context
from .history import get_conversation_history
from .models import AIModel

type ResponseType = list[str] | str | None

MAX_TELEGRAM_MESSAGE_LENGTH = 4096
CONVERSATION_HISTORY_LIMIT = 30
MODEL_PREFIX_LENGTH_CHECK = 10
DEFAULT_SEARCH_CUTOFF = 0.6
TRUNCATE_TOKEN_DIVISOR = 2

RE_MODEL = re.compile(r"use:\s*(\S+)", flags=re.IGNORECASE)
RE_CLEAN = re.compile(r"use:\s*\S+", flags=re.IGNORECASE)
RE_THINK = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
RE_INSTRUCTION = re.compile(r"^_instruction:.*?$", flags=re.MULTILINE)
RE_SESSION = re.compile(r"^_session:.*?$", flags=re.MULTILINE)
RE_MARKDOWN_DIVIDER = re.compile(r"\n\s*[-*_]{3,}\s*\n", flags=re.MULTILINE)
RE_MULTI_NEWLINES = re.compile(r"\n{3,}", flags=re.MULTILINE)
RE_MODEL_PREFIX = re.compile(r"^\[✨\s+([^\]]+)\]\s+")

MODEL_MAPPING: dict[str, AIModel] = {model.value.lower(): model for model in AIModel}
MODEL_ALIAS_MAPPING: dict[str, AIModel] = {
    # OpenAI aliases
    "gpt": AIModel.GPT_4_1,
    "mini": AIModel.GPT_4_1_MINI,
    "nano": AIModel.GPT_4_1_NANO,
    "o3": AIModel.O3,
    "o4-mini": AIModel.O4_MINI,
    # DeepSeek aliases
    "deepseek": AIModel.DEEPSEEK_V3,
    "deepseek-v3": AIModel.DEEPSEEK_V3,
    "deepseekv3": AIModel.DEEPSEEK_V3,
    "v3": AIModel.DEEPSEEK_V3,
    "deepseekr1": AIModel.DEEPSEEK_R1,
    "r1": AIModel.DEEPSEEK_R1,
    # Microsoft alias
    "mai": AIModel.MAI_DS_R1,
    # xAI aliases
    "grok": AIModel.GROK_3,
    "grok-mini": AIModel.GROK_3_MINI,
}

ERROR_MARKERS: list[str] = ["(content_filter)", "(RateLimitReached)", "(tokens_limit_reached)"]


async def _get_media_file(message: Message) -> File | None:
    """Retrieve a media file from a message.

    Extracts file objects from photos, stickers, or image documents in messages.
    Prioritizes the highest quality photo if multiple sizes are available.

    Args:
        message: The message containing potential media files

    Returns:
        File object if media is found, None otherwise
    """
    if not message.bot:
        return None

    if message.photo:
        return await message.bot.get_file(message.photo[-1].file_id)

    if message.sticker:
        return await message.bot.get_file(message.sticker.file_id)

    if (
        message.document
        and message.document.mime_type
        and message.document.mime_type.startswith("image/")
    ):
        return await message.bot.get_file(message.document.file_id)

    return None


def is_media_message(message: Message) -> bool:
    """Check if a message contains supported media content.

    Supported media types: photos, stickers, and image documents.

    Args:
        message: The message to check

    Returns:
        True if message contains supported media, False otherwise
    """
    return bool(
        message.photo
        or message.sticker
        or (
            message.document
            and message.document.mime_type
            and message.document.mime_type.startswith("image/")
        ),
    )


def find_best_model_match(model_name: str) -> AIModel:
    """Find the best matching model from a partial or alias name.

    Performs exact matching first, then fuzzy matching with aliases,
    and finally falls back to the default model.

    Args:
        model_name: Partial or full model name provided by user

    Returns:
        The matching AIModel or DEFAULT_MODEL if no match found
    """
    normalized_name = model_name.lower().strip()

    # Exact match in model mapping
    if normalized_name in MODEL_MAPPING:
        return MODEL_MAPPING[normalized_name]

    # Exact match in alias mapping
    if normalized_name in MODEL_ALIAS_MAPPING:
        return MODEL_ALIAS_MAPPING[normalized_name]

    # Fuzzy matching for model names
    model_matches = get_close_matches(
        normalized_name,
        list(MODEL_MAPPING.keys()),
        n=1,
        cutoff=DEFAULT_SEARCH_CUTOFF,
    )
    if model_matches:
        return MODEL_MAPPING[model_matches[0]]

    # Fuzzy matching for aliases
    alias_matches = get_close_matches(
        normalized_name,
        list(MODEL_ALIAS_MAPPING.keys()),
        n=1,
        cutoff=DEFAULT_SEARCH_CUTOFF,
    )
    if alias_matches:
        return MODEL_ALIAS_MAPPING[alias_matches[0]]

    return DEFAULT_MODEL


def parse_and_get_model(text: str | None) -> tuple[str, AIModel]:
    """Parse text for model specification and return cleaned text with selected model.

    Extracts model directives in the format "use: model_name" and removes them
    from the text while returning the corresponding model.

    Args:
        text: Input text that may include a model directive

    Returns:
        Tuple of (cleaned text, selected model)
    """
    if not text:
        return "", DEFAULT_MODEL

    model_match = RE_MODEL.search(text)
    if model_match:
        model_name = model_match.group(1)
        model = find_best_model_match(model_name)
    else:
        model = DEFAULT_MODEL

    clean_text = RE_CLEAN.sub("", text).strip()
    return clean_text, model


def clean_error_message(message: str) -> str:
    """Clean error messages by removing specific markers and extracting the first line.

    Args:
        message: Error message to clean

    Returns:
        Cleaned error message with markers removed
    """
    for marker in ERROR_MARKERS:
        if marker in message:
            return message.split("\n", 1)[0].replace(marker, "").strip()
    return message


def clean_response_output(response: str) -> str:
    """Clean response by removing internal instructions, think blocks, and formatting.

    Removes session information, instructions, think blocks, markdown dividers,
    and excessive newlines from the response.

    Args:
        response: Raw bot response text

    Returns:
        Cleaned response suitable for user display
    """
    cleaned = RE_SESSION.sub(
        "",
        RE_INSTRUCTION.sub("", RE_THINK.sub("", response).strip()).strip(),
    ).strip()

    return clean_and_format_llm_text(cleaned)


async def _download_media_file(message: Message, file_obj: File) -> Path:
    """Download media file to temporary location.

    Args:
        message: The message containing the media
        file_obj: File object from Telegram API

    Returns:
        Path to the downloaded file

    Raises:
        ValueError: If file path is None or bot is None
    """
    if not message.bot:
        error_msg = "Bot instance is None"
        raise ValueError(error_msg)

    if not file_obj.file_path:
        error_msg = "File path is None"
        raise ValueError(error_msg)

    local_filename = Path(tempfile.gettempdir()) / Path(file_obj.file_path).name
    await message.bot.download_file(file_obj.file_path, destination=local_filename)
    return local_filename


async def _process_media_with_ai(
    local_filename: Path,
    prompt: str,
    user: Any,
    model: AIModel,
    chat_history: list[Any],
) -> tuple[str, AIModel]:
    """Process media file with AI and return response.

    Args:
        local_filename: Path to the local media file
        prompt: Text prompt for the AI
        user: User object
        model: AI model to use
        chat_history: Conversation history

    Returns:
        Tuple of (response text, used model)

    Raises:
        HttpResponseError: If AI processing fails
    """
    return await query_azure_chat_with_image(
        str(local_filename),
        prompt,
        user,
        model,
        chat_history,
    )


async def _send_chunked_response(
    message: Message,
    response: str,
    used_model: AIModel,
    reply_to: Message,
) -> None:
    """Send response in chunks with model prefix.

    Args:
        message: Original message
        response: Response text to send
        used_model: AI model used
        reply_to: Message to reply to
    """
    chunks = split_text_with_formatting(response)
    if chunks:
        chunks[0] = f"[✨ {used_model.value}] {chunks[0]}"

    for chunk in chunks:
        await message.answer(telegram_format(chunk), reply_to_message_id=reply_to.message_id)


async def process_media_message(
    message: Message,
    target_message: Message,
    reply_to: Message,
) -> None:
    """Process messages with media content using Azure's image processing capabilities.

    Downloads media files, sends them to Azure API with any caption text,
    and responds with the AI-generated reply. Maintains conversation history.

    Args:
        message: The triggering message
        target_message: The message containing media
        reply_to: The message to reply to
    """
    if not is_media_message(target_message) or not message.from_user:
        return

    query_text = target_message.caption
    clean_text, model = parse_and_get_model(query_text)

    file_obj = await _get_media_file(target_message)
    if not file_obj:
        await message.answer("Media file not found.", reply_to_message_id=reply_to.message_id)
        return

    try:
        local_filename = await _download_media_file(target_message, file_obj)
        user_id = message.from_user.id
        chat_history = await get_conversation_history(user_id)
        image_caption = clean_text or "[Image without caption]"

        if reply_to and reply_to != target_message:
            updated_prompt, chat_history = build_reply_context(
                message,
                image_caption,
                chat_history,
            )
        else:
            updated_prompt = image_caption

        response, used_model = await _process_media_with_ai(
            local_filename,
            updated_prompt,
            message.from_user,
            model,
            chat_history,
        )

        if not response:
            await message.answer(
                "Could not generate a response.",
                reply_to_message_id=reply_to.message_id,
            )
            return

        clean_response = clean_response_output(response)
        await _send_chunked_response(message, clean_response, used_model, reply_to)
        await save_message(user_id, image_caption, clean_response)

    except HttpResponseError as chat_err:
        error_message = clean_error_message(chat_err.message)
        await message.answer(error_message, reply_to_message_id=reply_to.message_id)


def _determine_target_and_reply_messages(message: Message) -> tuple[Message, Message]:
    """Determine target message and reply-to message based on message context.

    Args:
        message: The incoming message

    Returns:
        Tuple of (target_message, reply_to_message)
    """
    target_message = (
        message.reply_to_message
        if message.reply_to_message and is_media_message(message.reply_to_message)
        else message
    )
    reply_to_message = message.reply_to_message or message
    return target_message, reply_to_message


async def _handle_text_message_processing(
    message: Message,
    reply_to_message: Message,
) -> None:
    """Handle processing of text messages.

    Args:
        message: The message to process
        reply_to_message: Message to reply to
    """
    text = message.text or message.caption
    if not text:
        return

    clean_text, model = parse_and_get_model(text)
    updated_message = message.model_copy(update={"text": clean_text})

    response = await process_message(updated_message, model)
    if not response:
        await message.answer(
            "Could not generate a response.",
            reply_to_message_id=reply_to_message.message_id,
        )
        return

    clean_response = clean_response_output(
        "".join(response) if isinstance(response, list) else response,
    )

    if any(marker in clean_response for marker in ERROR_MARKERS):
        await message.answer(clean_response, reply_to_message_id=reply_to_message.message_id)
        return

    chunks = split_text_with_formatting(clean_response)
    if chunks and "[✨" not in chunks[0][:MODEL_PREFIX_LENGTH_CHECK]:
        chunks[0] = f"[✨ {model.value}] {chunks[0]}"

    for chunk in chunks:
        await message.answer(
            telegram_format(chunk),
            reply_to_message_id=reply_to_message.message_id,
        )


async def process_and_reply(message: Message, *, clear: bool = False) -> None:
    """Process a message and generate an AI response.

    Handles both text and media messages, with optional history clearing.
    If the message is a reply to another message, the bot will respond to that message.

    Args:
        message: The message to process
        clear: Whether to clear conversation history before processing
    """
    if not message.from_user:
        return

    if clear:
        await clear_user_conversation_history(message.from_user.id)

    target_message, reply_to_message = _determine_target_and_reply_messages(message)

    if is_media_message(target_message):
        await process_media_message(message, target_message, reply_to=reply_to_message)
        return

    await _handle_text_message_processing(message, reply_to_message)


def _extract_model_prefix_and_content(response: str) -> tuple[str, str] | None:
    """Extract model prefix and content from response.

    Args:
        response: Full response string

    Returns:
        Tuple of (prefix, content) or None if no prefix found
    """
    model_prefix_match = RE_MODEL_PREFIX.match(response)
    if model_prefix_match:
        prefix = model_prefix_match.group(0)
        content = response[len(prefix) :]
        return prefix, content
    return None


def _split_long_response(response: str) -> list[str]:
    """Split long response into chunks while preserving formatting.

    Args:
        response: Response text to split

    Returns:
        List of text chunks
    """
    prefix_and_content = _extract_model_prefix_and_content(response)
    if prefix_and_content:
        prefix, content = prefix_and_content
        chunks = split_text_with_formatting(content)
        if chunks:
            chunks[0] = f"{prefix}{chunks[0]}"
        return chunks
    return split_text_with_formatting(response)


async def process_message(message: Message, model: AIModel) -> ResponseType:
    """Process a text message by building context and querying the AI model.

    Args:
        message: The incoming message with text
        model: The AI model to use

    Returns:
        Formatted response or None if processing failed
    """
    text_content = message.text or message.caption
    if not text_content or not text_content.strip():
        return None

    if not message.from_user:
        return None

    user_id = message.from_user.id
    chat_history = await get_conversation_history(user_id)
    updated_prompt, chat_history = build_reply_context(
        message,
        text_content.strip(),
        chat_history,
    )
    conversation_context = [*chat_history, UserMessage(content=updated_prompt)]

    try:
        reply_text, used_model = await query_azure_chat(
            messages=conversation_context,
            user=message.from_user,
            model=model,
        )
        full_response = f"[✨ {used_model.value}] {reply_text}"
    except HttpResponseError as chat_err:
        return clean_error_message(chat_err.message)

    await save_message(user_id, text_content.strip(), reply_text)

    if len(full_response) > MAX_TELEGRAM_MESSAGE_LENGTH:
        return _split_long_response(full_response)

    return full_response


async def save_message(user_id: int, user_message: str, bot_response: str) -> None:
    """Save conversation exchange to database and prune history if needed.

    Maintains a rolling history of the last N messages per user to prevent
    database bloat while preserving recent context.

    Args:
        user_id: User identifier
        user_message: User's message text
        bot_response: Bot's response text
    """
    await save_conversation(user_id, user_message, bot_response)
    await prune_conversation_history(user_id, keep_count=CONVERSATION_HISTORY_LIMIT)


def _split_code_blocks(text: str) -> list[str]:
    """Split text into segments based on Markdown-style code blocks.

    Separates input text into code blocks (delimited by triple backticks or tildes)
    and regular text segments, preserving their original order.

    Args:
        text: Input string potentially containing Markdown code blocks

    Returns:
        List of strings, each representing either a code block (including delimiters)
        or a regular text segment
    """
    parts: list[str] = []
    is_code_block = False
    code_block_buffer: list[str] = []
    current_buffer: list[str] = []
    code_block_delimiter: str | None = None

    for line in text.split("\n"):
        stripped = line.strip()

        if stripped.startswith(("```", "~~~")):
            delimiter = stripped[:3]

            if not is_code_block:
                if current_buffer:
                    parts.append("\n".join(current_buffer))
                    current_buffer = []
                code_block_buffer = [line]
                is_code_block = True
                code_block_delimiter = delimiter
            elif code_block_delimiter == delimiter:
                code_block_buffer.append(line)
                parts.append("\n".join(code_block_buffer))
                code_block_buffer = []
                is_code_block = False
                code_block_delimiter = None
            else:
                code_block_buffer.append(line)
        elif is_code_block:
            code_block_buffer.append(line)
        else:
            current_buffer.append(line)

    if current_buffer:
        parts.append("\n".join(current_buffer))
    if code_block_buffer:
        parts.append("\n".join(code_block_buffer))

    return parts


def clean_and_format_llm_text(text: str) -> str:
    """Clean and format text generated by a language model.

    Splits text into code and non-code blocks, removes excessive newlines and
    markdown dividers, while preserving code block formatting.

    Args:
        text: Raw text output from the LLM to be cleaned and formatted

    Returns:
        Cleaned and formatted text with code blocks preserved and non-code
        sections normalized
    """
    parts = _split_code_blocks(text)
    cleaned_parts: list[str] = []

    for part in parts:
        stripped = part.strip()
        if stripped.startswith(("```", "~~~")):
            cleaned_parts.append(part)
            continue

        cleaned_part = RE_MARKDOWN_DIVIDER.sub("\n\n", part)
        cleaned_part = RE_MULTI_NEWLINES.sub("\n\n", cleaned_part)
        cleaned_part = cleaned_part.replace("\r", "").strip()

        if cleaned_part:
            cleaned_parts.append(cleaned_part)

    return "\n\n".join(cleaned_parts)
