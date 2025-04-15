# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

import logging

import tiktoken
from azure.ai.inference.models import (
    AssistantMessage,
    ChatRequestMessage,
    TextContentItem,
    UserMessage,
)

from bot import config
from bot.utils.chat.models import AIModel

logger = logging.getLogger(__name__)

INPUT_TOKEN_LIMIT = config.token_truncate_limit


def get_token_encoding(model: AIModel):
    """
    Get the appropriate token encoding for a model.

    Args:
        model: AI model to determine encoding

    Returns:
        Tokenizer encoding
    """
    encoding_name = model.value if model != AIModel.GPT_4_1 else "gpt-4o"
    try:
        return tiktoken.encoding_for_model(encoding_name)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")  # Default fallback encoding


def truncate_content_by_tokens(content: str, model: AIModel, max_tokens: int) -> str:
    """
    Truncate content to a maximum token count.

    Args:
        content: Text content to truncate
        model: AI model to determine token encoding
        max_tokens: Maximum number of tokens to keep

    Returns:
        Truncated content that fits within token limit
    """
    enc = get_token_encoding(model)
    tokens = enc.encode(content)
    if len(tokens) <= max_tokens:
        return content

    logger.info(
        "Truncating content from %d to %d tokens for model %s",
        len(tokens),
        max_tokens,
        model.value,
    )
    return enc.decode(tokens[:max_tokens])


def get_message_token_count(message: ChatRequestMessage, model: AIModel) -> int:
    """
    Calculate token count for a message.

    Args:
        message: The message to calculate tokens for
        model: AI model to determine token encoding

    Returns:
        Approximate token count
    """
    enc = get_token_encoding(model)

    text_content = ""
    if isinstance(message, UserMessage) and message.content:
        for item in message.content:
            if isinstance(item, TextContentItem):
                text_content += item.text or ""
    elif (
        isinstance(message, UserMessage)
        and hasattr(message, "content")
        and isinstance(message.content, str)
    ):
        text_content = message.content

    return len(enc.encode(text_content))


def extract_newest_messages(messages: list[ChatRequestMessage]) -> list[ChatRequestMessage]:
    """
    Extract the most recent messages that should be preserved.

    Args:
        messages: List of conversation messages excluding system message

    Returns:
        List of newest messages to preserve
    """
    if not messages:
        return []

    newest = [messages[-1]]

    if len(messages) >= 2 and isinstance(messages[-2], AssistantMessage):
        newest.insert(0, messages[-2])

    return newest


def keep_messages_within_limit(
    reversed_messages: list[ChatRequestMessage], available_tokens: int, model: AIModel
) -> list[ChatRequestMessage]:
    """
    Process messages to fit within token limit.

    Args:
        reversed_messages: Messages in reverse chronological order
        available_tokens: Number of tokens available for these messages
        model: AI model to use for token calculation

    Returns:
        List of messages that fit within the token limit, in correct chronological order
    """
    kept_messages = []

    for message in reversed_messages:
        message_tokens = get_message_token_count(message, model)

        if message_tokens <= available_tokens:
            kept_messages.append(message)
            available_tokens -= message_tokens
        elif available_tokens > 0:
            truncated_message = try_truncate_message(message, available_tokens, model)
            if truncated_message:
                kept_messages.append(truncated_message)
            available_tokens = 0
            break

        if available_tokens <= 0:
            break

    kept_messages.reverse()
    return kept_messages


def try_truncate_message(
    message: ChatRequestMessage, available_tokens: int, model: AIModel
) -> ChatRequestMessage | None:
    """
    Attempt to truncate a message to fit within available tokens.

    Args:
        message: Message to truncate
        available_tokens: Number of tokens available
        model: AI model to use for token calculation

    Returns:
        Truncated message or None if truncation not possible
    """
    truncate_limit = available_tokens - 3
    if truncate_limit <= 0:
        return None

    if isinstance(message, UserMessage) and isinstance(message.content, list):
        for i, item in enumerate(message.content):
            if isinstance(item, TextContentItem) and item.text:
                truncated_text = truncate_content_by_tokens(item.text, model, truncate_limit)
                if truncated_text != item.text:
                    truncated_text += "..."
                    new_content = list(message.content)
                    new_content[i] = TextContentItem(text=truncated_text)
                    return UserMessage(content=new_content)

    elif isinstance(message, (UserMessage, AssistantMessage)) and isinstance(message.content, str):
        truncated_content = truncate_content_by_tokens(message.content, model, truncate_limit)
        if truncated_content != message.content:
            truncated_content += "..."
            if isinstance(message, AssistantMessage):
                return AssistantMessage(content=truncated_content)
            return UserMessage(content=truncated_content)

    return None


def truncate_messages(
    messages: list[ChatRequestMessage], model: AIModel
) -> list[ChatRequestMessage]:
    """
    Truncate messages to fit within token limits.

    This preserves the system message and recent messages while truncating older
    user/assistant messages.

    Args:
        messages: List of messages in the conversation
        model: AI model to use for token calculation

    Returns:
        Truncated list of messages that fits within the token limit
    """
    messages_copy = list(messages)

    if not messages_copy:
        return messages_copy

    system_message = messages_copy[0]

    if len(messages_copy) <= 3:
        return messages_copy

    remaining_messages = messages_copy[1:]

    system_tokens = get_message_token_count(system_message, model)
    available_tokens = INPUT_TOKEN_LIMIT - system_tokens

    newest_messages = extract_newest_messages(remaining_messages)
    newest_tokens = sum(get_message_token_count(msg, model) for msg in newest_messages)
    available_tokens -= newest_tokens

    kept_messages = keep_messages_within_limit(
        list(reversed(remaining_messages[: -len(newest_messages)])), available_tokens, model
    )

    truncated_messages = [system_message, *kept_messages, *newest_messages]

    if len(truncated_messages) < len(messages):
        logger.info(
            "Truncated conversation from %d to %d messages to fit token limit",
            len(messages),
            len(truncated_messages),
        )

    return truncated_messages
