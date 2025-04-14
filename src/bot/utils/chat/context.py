# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

from enum import StrEnum
from typing import TypedDict

from aiogram.types import Message
from azure.ai.inference.models import AssistantMessage, UserMessage

type HistoryType = list[AssistantMessage | UserMessage]


class RelationshipType(StrEnum):
    """Enumeration for message relationship types."""

    BOT = "bot"
    SELF = "self"
    OTHER = "other"


class Relationship(TypedDict):
    """Dictionary type for message relationship information."""

    kind: RelationshipType
    note: str


def build_reply_context(
    message: Message, user_prompt: str, history: HistoryType
) -> tuple[str, HistoryType]:
    """
    Build reply context from a message that may contain a reply.

    Incorporates the replied message content into the user prompt and updates
    the conversation history. This enriches the context for the AI model to
    generate more relevant responses.

    Args:
        message: Message that may contain a reply
        user_prompt: Original user input prompt
        history: Existing conversation history

    Returns:
        Tuple of (updated prompt with context, updated history list)
    """
    reply_msg = message.reply_to_message
    if not reply_msg or not reply_msg.from_user:
        return user_prompt.strip(), history

    content = extract_message_content(reply_msg)
    if not content:
        return user_prompt.strip(), history

    sender = reply_msg.from_user
    sender_name = sender.full_name.strip()

    relationship = determine_message_relationship(message, reply_msg)
    if not relationship:
        return user_prompt.strip(), history

    context_parts = [f"_instruction: {relationship['note']} Use the replied message as context."]

    if relationship["kind"] == RelationshipType.BOT:
        context_parts.append(f"My previous response: '{content}'")
    elif relationship["kind"] == RelationshipType.SELF:
        context_parts.append(f"User's previous message: '{content}'")
    else:
        context_parts.extend([
            f"Replied message content: '{content}'",
            f"Sent by: {sender_name}",
        ])

    entities = extract_message_entities(reply_msg)
    if entities:
        context_parts.append(f"Message contains: {entities}")

    context = "\n".join(context_parts)
    updated_prompt = f"{context}\nUser Prompt: {user_prompt.strip()}"

    history.append(UserMessage(content=updated_prompt))
    return updated_prompt, history


def _format_media(label: str, message: Message) -> str:
    """
    Formats a media attachment label and its caption into a string.

    Args:
        label (str): A label describing the type of media (e.g., "Image", "Video").
        message (Message): The message object containing the media and its optional caption.

    Returns:
        str: A formatted string indicating the media type and its caption, if present.
    """
    caption_text = f": {message.caption.strip()}" if message.caption else ""
    return f"[{label} attached]{caption_text}"


def extract_message_content(message: Message) -> str | None:
    """
    Extract content from various message types.

    Handles different Telegram message types (text, photo, video, etc.) and
    returns a string representation of the content.

    Args:
        message: Telegram message to extract content from

    Returns:
        String representation of the message content or None if no content
    """
    if content := _extract_text_or_caption(message):
        return content
    if content := _extract_media_content(message):
        return content
    if message.voice:
        return "[Voice message attached]"
    if message.document:
        return f"[Document attached: {message.document.file_name}]"
    if message.sticker:
        return _extract_sticker_content(message)
    if message.poll:
        return f"[Poll: {message.poll.question}]"
    if message.location:
        return "[Location shared]"
    return None


def _extract_text_or_caption(message: Message) -> str | None:
    """
    Extracts and returns the text or caption from a given message.

    If the message contains text, it returns the stripped text. If the message
    contains a caption, it returns the stripped caption. If neither is present,
    it returns None.

    Args:
        message (Message): The message object containing text or caption.

    Returns:
        str | None: The stripped text or caption if available, otherwise None.
    """
    if message.text:
        return message.text.strip()
    if message.caption:
        return message.caption.strip()
    return None


def _extract_media_content(message: Message) -> str | None:
    """
    Extract media content from a message.

    This function checks the provided message for various types of media content
    such as photos, videos, audio, or animations. If a specific type of media is
    found, it formats the media information using the `_format_media` helper
    function and returns it as a string. If no media is present in the message,
    the function returns `None`.

    Args:
        message (Message): The message object to extract media content from.

    Returns:
        str | None: A formatted string describing the media content if present,
        otherwise `None`.
    """
    if message.photo:
        return _format_media("Photo", message)
    if message.video:
        return _format_media("Video", message)
    if message.audio:
        return _format_media("Audio", message)
    if message.animation:
        return _format_media("Animation", message)
    return None


def _extract_sticker_content(message: Message) -> str:
    """
    Extracts and formats the content of a sticker from a given message.

    If the message contains a sticker with an associated emoji, the emoji is included
    in the returned string. If no emoji is available, a default message is used.

    Args:
        message (Message): The message object containing the sticker.

    Returns:
        str: A formatted string representing the sticker content, including its emoji
             or a default message if the emoji is not available.
    """
    emoji = (
        message.sticker.emoji
        if message.sticker and message.sticker.emoji
        else "emoji not available"
    )
    return f"[Sticker: {emoji}]"


def extract_message_entities(message: Message) -> str | None:
    """
    Extract and describe entities from a message.

    Identifies URLs, mentions, hashtags, code snippets and blocks in the message.

    Args:
        message: Message to extract entities from

    Returns:
        Comma-separated string of entity types or None if no entities
    """
    if not message.entities:
        return None

    entity_descriptions = {
        "url": "URL",
        "mention": "User mention",
        "hashtag": "Hashtag",
        "code": "Code snippet",
        "pre": "Code block",
    }

    entity_types = [
        description
        for entity in message.entities
        if (description := entity_descriptions.get(entity.type))
    ]

    return ", ".join(entity_types) if entity_types else None


def determine_message_relationship(message: Message, reply_msg: Message) -> Relationship | None:
    """
    Determine relationship between message sender and replied message sender.

    Identifies if the user is replying to the bot, themselves, or another user.

    Args:
        message: Current message
        reply_msg: Message being replied to

    Returns:
        Dictionary with relationship type and descriptive note, or None
    """
    if not reply_msg.from_user:
        return None

    if message.bot and reply_msg.from_user.id == message.bot.id:
        return {"kind": RelationshipType.BOT, "note": "User replied to my previous message."}

    if message.from_user and reply_msg.from_user.id == message.from_user.id:
        return {"kind": RelationshipType.SELF, "note": "User replied to their own message."}

    return {"kind": RelationshipType.OTHER, "note": "User replied to another user's message."}
