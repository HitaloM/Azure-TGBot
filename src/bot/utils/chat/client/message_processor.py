# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import tiktoken
from azure.ai.inference.models import (
    AssistantMessage,
    ChatRequestMessage,
    TextContentItem,
    UserMessage,
)

from bot import config
from bot.utils.chat.models import AIModel

if TYPE_CHECKING:
    import tiktoken
    from azure.ai.inference.models import (
        AssistantMessage,
        ChatRequestMessage,
        TextContentItem,
        UserMessage,
    )

    from bot.utils.chat.models import AIModel

logger = logging.getLogger(__name__)

INPUT_TOKEN_LIMIT = config.token_truncate_limit
DEFAULT_ENCODING = "cl100k_base"
GPT_4O_MODEL_NAME = "gpt-4o"
TRUNCATION_SUFFIX = "[...]"
TRUNCATION_BUFFER_TOKENS = 3
MINIMUM_CONVERSATION_LENGTH = 3


def _get_model_encoding_name(model: AIModel) -> str:
    """
    Get the appropriate encoding name for a specific model.

    Args:
        model: AI model to determine encoding name

    Returns:
        Encoding name string for the model
    """
    if model == AIModel.GPT_4_1:
        return GPT_4O_MODEL_NAME
    return model.value


def _get_fallback_encoding() -> tiktoken.Encoding:
    """
    Get the default fallback encoding when model-specific encoding fails.

    Returns:
        Default tiktoken encoding
    """
    return tiktoken.get_encoding(DEFAULT_ENCODING)


def get_token_encoding(model: AIModel) -> tiktoken.Encoding:
    """
    Get the appropriate token encoding for a model.

    Args:
        model: AI model to determine encoding

    Returns:
        Tokenizer encoding
    """
    encoding_name = _get_model_encoding_name(model)
    try:
        return tiktoken.encoding_for_model(encoding_name)
    except KeyError:
        return _get_fallback_encoding()


def _log_truncation_info(original_tokens: int, max_tokens: int, model: AIModel) -> None:
    """
    Log information about content truncation.

    Args:
        original_tokens: Original token count
        max_tokens: Maximum allowed tokens
        model: AI model being used
    """
    logger.info(
        "Truncating content from %d to %d tokens for model %s",
        original_tokens,
        max_tokens,
        model.value,
    )


def _encode_and_check_length(content: str, encoding: tiktoken.Encoding) -> tuple[list[int], int]:
    """
    Encode content and return tokens with count.

    Args:
        content: Text content to encode
        encoding: Tiktoken encoding to use

    Returns:
        Tuple of (encoded tokens, token count)
    """
    tokens = encoding.encode(content)
    return tokens, len(tokens)


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
    tokens, token_count = _encode_and_check_length(content, enc)

    if token_count <= max_tokens:
        return content

    _log_truncation_info(token_count, max_tokens, model)
    return enc.decode(tokens[:max_tokens])


def _extract_text_from_user_message_content(message: UserMessage) -> str:
    """
    Extract text content from a UserMessage with list content.

    Args:
        message: UserMessage with list-based content

    Returns:
        Concatenated text content from all TextContentItem elements
    """
    text_content = ""
    if message.content:
        for item in message.content:
            if isinstance(item, TextContentItem):
                text_content += item.text or ""
    return text_content


def _extract_text_from_string_content(message: UserMessage) -> str:
    """
    Extract text content from a UserMessage with string content.

    Args:
        message: UserMessage with string-based content

    Returns:
        String content of the message
    """
    if hasattr(message, "content") and isinstance(message.content, str):
        return message.content
    return ""


def _extract_message_text_content(message: ChatRequestMessage) -> str:
    """
    Extract text content from various message types.

    Args:
        message: Message to extract text from

    Returns:
        Text content from the message
    """
    if isinstance(message, UserMessage) and message.content:
        if isinstance(message.content, list):
            return _extract_text_from_user_message_content(message)
        return _extract_text_from_string_content(message)
    return ""


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
    text_content = _extract_message_text_content(message)
    return len(enc.encode(text_content))


def _get_last_message(messages: list[ChatRequestMessage]) -> list[ChatRequestMessage]:
    """
    Get the last message from the conversation.

    Args:
        messages: List of conversation messages

    Returns:
        List containing the last message
    """
    return [messages[-1]]


def _should_include_assistant_message(messages: list[ChatRequestMessage], index: int) -> bool:
    """
    Check if an assistant message should be included in newest messages.

    Args:
        messages: List of conversation messages
        index: Index of the potential assistant message

    Returns:
        True if the assistant message should be included
    """
    return len(messages) >= 2 and index >= 0 and isinstance(messages[index], AssistantMessage)


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

    newest = _get_last_message(messages)
    assistant_index = len(messages) - 2

    if _should_include_assistant_message(messages, assistant_index):
        newest.insert(0, messages[assistant_index])

    return newest


def _process_message_for_token_limit(
    message: ChatRequestMessage, available_tokens: int, model: AIModel
) -> tuple[ChatRequestMessage | None, int]:
    """
    Process a single message for token limit compliance.

    Args:
        message: Message to process
        available_tokens: Available token budget
        model: AI model for token calculation

    Returns:
        Tuple of (processed message or None, remaining tokens)
    """
    message_tokens = get_message_token_count(message, model)

    if message_tokens <= available_tokens:
        return message, available_tokens - message_tokens

    if available_tokens > 0:
        truncated_message = try_truncate_message(message, available_tokens, model)
        if truncated_message:
            return truncated_message, 0

    return None, 0


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
    remaining_tokens = available_tokens

    for message in reversed_messages:
        processed_message, remaining_tokens = _process_message_for_token_limit(
            message, remaining_tokens, model
        )

        if processed_message:
            kept_messages.append(processed_message)

        if remaining_tokens <= 0:
            break

    kept_messages.reverse()
    return kept_messages


def _calculate_truncation_limit(available_tokens: int) -> int:
    """
    Calculate the effective truncation limit accounting for buffer.

    Args:
        available_tokens: Available token budget

    Returns:
        Truncation limit with buffer applied
    """
    return available_tokens - TRUNCATION_BUFFER_TOKENS


def _truncate_user_message_list_content(
    message: UserMessage, truncate_limit: int, model: AIModel
) -> UserMessage | None:
    """
    Truncate a UserMessage with list-based content.

    Args:
        message: UserMessage with list content
        truncate_limit: Maximum tokens allowed
        model: AI model for token calculation

    Returns:
        Truncated UserMessage or None if not possible
    """
    if not isinstance(message.content, list):
        return None

    for i, item in enumerate(message.content):
        if isinstance(item, TextContentItem) and item.text:
            truncated_text = truncate_content_by_tokens(item.text, model, truncate_limit)
            if truncated_text != item.text:
                truncated_text += TRUNCATION_SUFFIX
                new_content = list(message.content)
                new_content[i] = TextContentItem(text=truncated_text)
                return UserMessage(content=new_content)

    return None


def _truncate_message_string_content(
    message: UserMessage | AssistantMessage, truncate_limit: int, model: AIModel
) -> UserMessage | AssistantMessage | None:
    """
    Truncate a message with string-based content.

    Args:
        message: Message with string content
        truncate_limit: Maximum tokens allowed
        model: AI model for token calculation

    Returns:
        Truncated message or None if not possible
    """
    if not isinstance(message.content, str):
        return None

    truncated_content = truncate_content_by_tokens(message.content, model, truncate_limit)
    if truncated_content != message.content:
        truncated_content += TRUNCATION_SUFFIX
        if isinstance(message, AssistantMessage):
            return AssistantMessage(content=truncated_content)
        return UserMessage(content=truncated_content)

    return None


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
    truncate_limit = _calculate_truncation_limit(available_tokens)
    if truncate_limit <= 0:
        return None

    if isinstance(message, UserMessage) and isinstance(message.content, list):
        return _truncate_user_message_list_content(message, truncate_limit, model)

    if isinstance(message, (UserMessage, AssistantMessage)) and isinstance(message.content, str):
        return _truncate_message_string_content(message, truncate_limit, model)

    return None


def _extract_system_and_remaining_messages(
    messages: list[ChatRequestMessage],
) -> tuple[ChatRequestMessage, list[ChatRequestMessage]]:
    """
    Extract system message and remaining messages from conversation.

    Args:
        messages: Complete list of conversation messages

    Returns:
        Tuple of (system message, remaining messages)
    """
    return messages[0], messages[1:]


def _calculate_available_tokens_for_history(
    system_message: ChatRequestMessage, newest_messages: list[ChatRequestMessage], model: AIModel
) -> int:
    """
    Calculate tokens available for historical messages.

    Args:
        system_message: The system message
        newest_messages: Messages to preserve
        model: AI model for token calculation

    Returns:
        Number of tokens available for historical messages
    """
    system_tokens = get_message_token_count(system_message, model)
    newest_tokens = sum(get_message_token_count(msg, model) for msg in newest_messages)
    return INPUT_TOKEN_LIMIT - system_tokens - newest_tokens


def _get_historical_messages(
    remaining_messages: list[ChatRequestMessage], newest_count: int
) -> list[ChatRequestMessage]:
    """
    Get historical messages excluding the newest ones.

    Args:
        remaining_messages: All messages except system message
        newest_count: Number of newest messages to exclude

    Returns:
        Historical messages in reverse order
    """
    if newest_count >= len(remaining_messages):
        return []
    return list(reversed(remaining_messages[:-newest_count]))


def _log_truncation_summary(original_count: int, final_count: int) -> None:
    """
    Log summary of message truncation.

    Args:
        original_count: Original message count
        final_count: Final message count after truncation
    """
    logger.info(
        "Truncated conversation from %d to %d messages to fit token limit",
        original_count,
        final_count,
    )


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

    if not messages_copy or len(messages_copy) <= MINIMUM_CONVERSATION_LENGTH:
        return messages_copy

    system_message, remaining_messages = _extract_system_and_remaining_messages(messages_copy)
    newest_messages = extract_newest_messages(remaining_messages)

    available_tokens = _calculate_available_tokens_for_history(
        system_message, newest_messages, model
    )

    historical_messages = _get_historical_messages(remaining_messages, len(newest_messages))
    kept_messages = keep_messages_within_limit(historical_messages, available_tokens, model)

    truncated_messages = [system_message, *kept_messages, *newest_messages]

    if len(truncated_messages) < len(messages):
        _log_truncation_summary(len(messages), len(truncated_messages))

    return truncated_messages
