# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

import re
import tempfile
from difflib import get_close_matches
from pathlib import Path

from aiogram.types import File, Message
from azure.ai.inference.models import UserMessage
from azure.core.exceptions import HttpResponseError
from chatgpt_md_converter import telegram_format

from bot.database.models import Conversation
from bot.utils.text_splitter import split_text_with_formatting

from .client import DEFAULT_MODEL, query_azure_chat, query_azure_chat_with_image
from .context import build_reply_context
from .history import clear_conversation_history, get_conversation_history
from .models import AIModel

type ResponseType = list[str] | str | None

RE_MODEL = re.compile(r"use:\s*(\S+)", flags=re.IGNORECASE)
RE_CLEAN = re.compile(r"use:\s*\S+", flags=re.IGNORECASE)
RE_THINK = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
RE_INSTRUCTION = re.compile(r"^_instruction:.*?$", flags=re.MULTILINE)
RE_SESSION = re.compile(r"^_session:.*?$", flags=re.MULTILINE)

MODEL_MAPPING = {model.value.lower(): model for model in AIModel}
MODEL_ALIAS_MAPPING = {
    # OpenAI aliases
    "gpt": AIModel.GPT_4_1,
    "4o": AIModel.GPT_4O_MINI,
    "mini": AIModel.GPT_4O_MINI,
    "o1": AIModel.O1_PREVIEW,
    "o3": AIModel.O3_MINI,
    # DeepSeek aliases
    "deepseek": AIModel.DEEPSEEK_V3,  # Default to V3 when just "deepseek" is specified
    "deepseek-v3": AIModel.DEEPSEEK_V3,
    "deepseekv3": AIModel.DEEPSEEK_V3,
    "v3": AIModel.DEEPSEEK_V3,
    "deepseekr1": AIModel.DEEPSEEK_R1,
    "r1": AIModel.DEEPSEEK_R1,
}

ERROR_MARKERS = ["(content_filter)", "(RateLimitReached)", "(tokens_limit_reached)"]


async def _get_media_file(message: Message) -> File | None:
    """
    Retrieve a media file from a message.

    Extracts file objects from photos, stickers, or image documents in messages.

    Args:
        message: The message containing potential media files

    Returns:
        The file object if found, None otherwise
    """
    if message.photo:
        return await message.bot.get_file(message.photo[-1].file_id)  # type: ignore
    if message.sticker:
        return await message.bot.get_file(message.sticker.file_id)  # type: ignore
    if (
        message.document
        and message.document.mime_type
        and message.document.mime_type.startswith("image/")
    ):
        return await message.bot.get_file(message.document.file_id)  # type: ignore
    return None


def is_media_message(message: Message) -> bool:
    """
    Check if a message contains media content (photo, sticker, or image document).

    Args:
        message: The message to check

    Returns:
        True if the message contains media, False otherwise
    """
    return bool(
        message.photo
        or message.sticker
        or (
            message.document
            and message.document.mime_type
            and message.document.mime_type.startswith("image/")
        )
    )


def find_best_model_match(model_name: str) -> AIModel:
    """
    Find the best matching model from a partial or alias name.

    Args:
        model_name: Partial or full model name provided by user

    Returns:
        The matching AIModel or DEFAULT_MODEL if no match found
    """
    model_name = model_name.lower().strip()

    if model_name in MODEL_MAPPING:
        return MODEL_MAPPING[model_name]

    if model_name in MODEL_ALIAS_MAPPING:
        return MODEL_ALIAS_MAPPING[model_name]

    all_model_names = list(MODEL_MAPPING.keys())
    close_matches = get_close_matches(model_name, all_model_names, n=1, cutoff=0.6)
    if close_matches:
        return MODEL_MAPPING[close_matches[0]]

    all_aliases = list(MODEL_ALIAS_MAPPING.keys())
    close_matches = get_close_matches(model_name, all_aliases, n=1, cutoff=0.6)
    if close_matches:
        return MODEL_ALIAS_MAPPING[close_matches[0]]

    return DEFAULT_MODEL


def parse_and_get_model(text: str | None) -> tuple[str, AIModel]:
    """
    Parse text for model specification and return cleaned text with model.

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


def clean_error_message(msg: str) -> str:
    """
    Clean error messages by removing specific markers and returning the first line.

    Args:
        msg: Error message to clean

    Returns:
        Cleaned error message
    """
    for marker in ERROR_MARKERS:
        if marker in msg:
            return msg.split("\n")[0].replace(marker, "").strip()
    return msg


def clean_response_output(response: str) -> str:
    """
    Clean response by removing internal instructions and think blocks.

    Args:
        response: Raw bot response text

    Returns:
        Cleaned response suitable for user display
    """
    return RE_SESSION.sub(
        "", RE_INSTRUCTION.sub("", RE_THINK.sub("", response).strip()).strip()
    ).strip()


async def process_media_message(
    message: Message, target_message: Message, reply_to: Message
) -> None:
    """
    Process messages with media content using Azure's image processing capabilities.

    Downloads media files, sends them to Azure API with any caption text,
    and responds with the AI-generated reply.

    Args:
        message: The triggering message
        target_message: The message containing media
        reply_to: The message to reply to
    """
    if not is_media_message(target_message):
        return

    query_text = target_message.caption
    clean_text, model = parse_and_get_model(query_text)

    file_obj = await _get_media_file(target_message)
    if not file_obj:
        await message.answer("Media file not found.", reply_to_message_id=reply_to.message_id)
        return

    local_filename = Path(tempfile.gettempdir()) / Path(file_obj.file_path).name  # type: ignore
    await target_message.bot.download_file(file_obj.file_path, destination=local_filename)  # type: ignore

    try:
        response, used_model = await query_azure_chat_with_image(
            str(local_filename),
            clean_text,
            message.from_user,  # type: ignore
            model,
        )
    except HttpResponseError as chat_err:
        await message.answer(
            clean_error_message(chat_err.message), reply_to_message_id=reply_to.message_id
        )
        return

    if not response:
        await message.answer(
            "Could not generate a response.", reply_to_message_id=reply_to.message_id
        )
        return

    clean_response = clean_response_output(response)
    full_response = f"[✨ {used_model.value}] {clean_response}"

    chunks = split_text_with_formatting(telegram_format(full_response))
    for chunk in chunks:
        await message.answer(chunk, reply_to_message_id=reply_to.message_id)

    await save_message(message.from_user.id, clean_text, clean_response)  # type: ignore


async def process_and_reply(message: Message, *, clear: bool = False) -> None:
    """
    Process a message and generate an AI response.

    Handles both text and media messages, with optional history clearing.
    If the message is a reply to another message, the bot will respond to that message.

    Args:
        message: The message to process
        clear: Whether to clear conversation history before processing
    """
    if not message.from_user:
        return

    if clear:
        await clear_conversation_history(message.from_user.id)

    target_message = (
        message.reply_to_message
        if message.reply_to_message and is_media_message(message.reply_to_message)
        else message
    )

    reply_to_message = message.reply_to_message or message

    if is_media_message(target_message):
        await process_media_message(message, target_message, reply_to=reply_to_message)
        return

    text = message.text or message.caption
    if not text:
        return

    clean_text, model = parse_and_get_model(text)
    updated_message = message.model_copy(update={"text": clean_text})

    response = await process_message(updated_message, model)
    if not response:
        await message.answer(
            "Could not generate a response.", reply_to_message_id=reply_to_message.message_id
        )
        return

    clean_response = clean_response_output(
        "".join(response) if isinstance(response, list) else response
    )

    chunks = split_text_with_formatting(telegram_format(clean_response))
    for chunk in chunks:
        await message.answer(chunk, reply_to_message_id=reply_to_message.message_id)


async def process_message(message: Message, model: AIModel) -> ResponseType:
    """
    Process a text message by building context and querying the AI model.

    Args:
        message: The incoming message with text
        model: The AI model to use

    Returns:
        Formatted response or None if processing failed
    """
    text_content = message.text or message.caption
    if not text_content or not text_content.strip():
        return None

    user_id = message.from_user.id  # type: ignore

    chat_history = await get_conversation_history(user_id)
    updated_prompt, chat_history = build_reply_context(message, text_content.strip(), chat_history)
    conversation_context = [*chat_history, UserMessage(content=updated_prompt)]

    try:
        reply_text, used_model = await query_azure_chat(
            messages=conversation_context,
            user=message.from_user,  # type: ignore
            model=model,
        )
        full_response = f"[✨ {used_model.value}] {reply_text}"
    except HttpResponseError as chat_err:
        return clean_error_message(chat_err.message)

    await save_message(user_id, text_content.strip(), reply_text)

    if len(full_response) > 4096:
        return split_text_with_formatting(telegram_format(full_response))

    return full_response


async def save_message(user_id: int, user_msg: str, bot_resp: str) -> None:
    """
    Save conversation exchange to database and prune history if needed.

    Maintains a rolling history of the last 30 messages per user.

    Args:
        user_id: User identifier
        user_msg: User's message text
        bot_resp: Bot's response text
    """
    await Conversation.create(user_id=user_id, user_message=user_msg, bot_response=bot_resp)

    records = await Conversation.filter(user_id=user_id).order_by("-timestamp").all()

    if len(records) > 30:
        ids_to_delete = [record.id for record in records[30:]]
        await Conversation.filter(id__in=ids_to_delete).delete()
