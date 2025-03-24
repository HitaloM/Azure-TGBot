# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

import re
import tempfile
from pathlib import Path

from aiogram.types import Message
from azure.ai.inference.models import UserMessage
from chatgpt_md_converter import telegram_format

from bot.database.models import Conversation

from .client import ChatAPIError, query_azure_chat, query_azure_chat_with_image
from .context import build_reply_context
from .history import clear_conversation_history, get_conversation_history
from .models import DEFAULT_MODEL, AIModel

RE_MODEL = re.compile(r"use:\s*(\S+)", flags=re.IGNORECASE)
RE_CLEAN = re.compile(r"use:\s*\S+|/\S+", flags=re.IGNORECASE)
RE_THINK = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)

MODEL_MAPPING = {model.value.lower(): model for model in AIModel}


def parse_and_get_model(text: str) -> tuple[str, AIModel]:
    """
    Parse the given text for an AI model specification and return a tuple containing
    the cleaned text and the corresponding AIModel.

    Args:
        text (str): The input text that may include a model directive.

    Returns:
        tuple[str, AIModel]: A tuple where the first element is the text with the model
            directive removed, and the second element is the determined AIModel.
    """
    model_match = RE_MODEL.search(text)
    model = (
        MODEL_MAPPING.get(model_match.group(1).lower(), DEFAULT_MODEL)
        if model_match
        else DEFAULT_MODEL
    )
    clean_text = RE_CLEAN.sub("", text).strip()
    return clean_text, model


def clean_error_message(msg: str) -> str:
    if "(content_filter)" in msg:
        return msg.split("\n")[0].replace("(content_filter)", "").strip()
    return msg


async def process_media_message(message: Message, target_message: Message) -> None:
    """
    Process a message that contains photo or sticker media and forward it to the
    Azure ChatCompletions API along with an optional text caption.

    This function retrieves the media file, downloads it locally, queries the API with the image,
    formats the response, and sends the reply back.

    Args:
        message (Message): The original message triggering the process.
        target_message (Message): The message containing media content.
    """
    if not (target_message.photo or target_message.sticker):
        return

    query_text = target_message.caption
    clean_text, model = parse_and_get_model(query_text) if query_text else ("", DEFAULT_MODEL)

    file_obj = None
    if target_message.photo:
        file_obj = await target_message.bot.get_file(target_message.photo[-1].file_id)  # type: ignore
    elif target_message.sticker:
        file_obj = await target_message.bot.get_file(target_message.sticker.file_id)  # type: ignore

    if not file_obj:
        await message.answer("No media file found.")
        return

    tmp_dir = tempfile.gettempdir()
    local_filename = Path(tmp_dir) / Path(file_obj.file_path).name  # type: ignore
    await target_message.bot.download_file(file_obj.file_path, destination=local_filename)  # type: ignore

    try:
        response = await query_azure_chat_with_image(
            str(local_filename),
            clean_text,
            message.from_user,  # type: ignore
            model,
        )
    except ChatAPIError as chat_err:
        await message.answer(clean_error_message(chat_err.message))
        return
    except Exception as error:
        await message.answer(str(error))
        return

    if response:
        clean_response = RE_THINK.sub("", response).strip()
        full_response = f"[✨ {model.value}] {clean_response}"
        await message.answer(telegram_format(full_response))
        await save_message(message.from_user.id, clean_text, clean_response)  # type: ignore
        return

    await message.answer("No response generated. :/")


async def process_and_reply(message: Message, *, clear: bool = False) -> None:
    """
    Process an incoming text message and reply using the Azure ChatCompletions API.

    If the message includes media content (photo or sticker), it delegates to
    process_media_message. Optionally clears the conversation history before processing.

    Args:
        message (Message): The incoming message from the user.
        clear (bool, optional): If True, clears the conversation history prior to processing.
            Defaults to False.
    """
    if not message.from_user:
        return

    if clear:
        await clear_conversation_history(message.from_user.id)

    target_message = (
        message.reply_to_message
        if (
            message.reply_to_message
            and (message.reply_to_message.photo or message.reply_to_message.sticker)
            and not (message.photo or message.sticker)
        )
        else message
    )

    if target_message.photo or target_message.sticker:
        await process_media_message(message, target_message)
        return

    text = message.text or message.caption
    if not text:
        return

    clean_text, model = parse_and_get_model(text)
    updated_message = message.model_copy(update={"text": clean_text})
    response = await process_message(updated_message, model)
    if response:
        clean_response = RE_THINK.sub("", response).strip()
        await message.answer(telegram_format(clean_response))
        return

    await message.answer("No response generated. :/")


async def process_message(message: Message, model: AIModel) -> str | None:
    """
    Process a text message by building the conversation context, querying the
    Azure ChatCompletions API, and saving the exchange.

    Args:
        message (Message): The incoming message containing text.
        model (AIModel): The AI model to be used for generating the reply.

    Returns:
        str | None: The full formatted response if successful, or None if the message is empty.
    """
    input_text = (message.text or message.caption).strip()  # type: ignore
    if not input_text:
        return None

    user_id = message.from_user.id  # type: ignore
    chat_history = await get_conversation_history(user_id)
    updated_prompt, chat_history = build_reply_context(message, input_text, chat_history)
    conversation_context = [*chat_history, UserMessage(content=updated_prompt)]

    try:
        reply_text = await query_azure_chat(
            messages=conversation_context,
            user=message.from_user,  # type: ignore
            model=model,
        )
        full_response = f"[✨ {model.value}] {reply_text}"
    except ChatAPIError as chat_err:
        return clean_error_message(chat_err.message)
    except Exception as error:
        return str(error)

    await save_message(user_id, input_text, reply_text)
    return full_response


async def save_message(user_id: int, user_msg: str, bot_resp: str):
    """
    Save an exchange to the conversation database and prune the history if necessary.

    Args:
        user_id (int): The unique identifier of the user.
        user_msg (str): The message text sent by the user.
        bot_resp (str): The response text generated by the bot.
    """
    await Conversation.create(user_id=user_id, user_message=user_msg, bot_response=bot_resp)
    records = await Conversation.filter(user_id=user_id).order_by("-timestamp").all()
    if len(records) > 30:
        ids_to_delete = [record.id for record in records[30:]]
        await Conversation.filter(id__in=ids_to_delete).delete()
