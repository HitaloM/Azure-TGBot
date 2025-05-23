# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from aiogram.types import User
from azure.ai.inference.aio import ChatCompletionsClient
from azure.ai.inference.models import (
    ChatCompletionsToolChoicePreset,
    ChatRequestMessage,
    ChatResponseMessage,
    ImageContentItem,
    ImageDetailLevel,
    ImageUrl,
    TextContentItem,
    UserMessage,
)
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

from bot import config
from bot.utils.chat.history import ChatHistory
from bot.utils.chat.models import AIModel
from bot.utils.chat.system_message import get_system_message
from bot.utils.chat.tools.tool_manager import tool_manager

from .message_processor import truncate_messages
from .rate_limiter import rate_limit_tracker
from .retry_policy import CustomRetryPolicy
from .tool_handler import process_tool_calls
from .utils import extract_retry_seconds_from_error

if TYPE_CHECKING:
    from aiogram.types import User
    from azure.ai.inference.aio import ChatCompletionsClient
    from azure.ai.inference.models import (
        ChatCompletionsToolChoicePreset,
        ChatRequestMessage,
        ChatResponseMessage,
        ImageContentItem,
        ImageDetailLevel,
        ImageUrl,
        TextContentItem,
        UserMessage,
    )
    from azure.core.credentials import AzureKeyCredential
    from azure.core.exceptions import HttpResponseError

    from bot.utils.chat.history import ChatHistory
    from bot.utils.chat.models import AIModel

logger = logging.getLogger(__name__)

ModelResponse = tuple[str, AIModel]

AZURE_API_VERSION = "2025-03-01-preview"
DEFAULT_IMAGE_EXTENSION = "jpg"
RATE_LIMIT_HTTP_STATUS = 429
EMPTY_RESPONSE_ERROR_MESSAGE = "Azure ChatCompletions API returned an empty response."

DEFAULT_MODEL = AIModel.GPT_4_1
FALLBACK_MODEL = AIModel.GPT_4_1_MINI

IMAGE_SUPPORTED_MODELS: set[AIModel] = {
    AIModel.GPT_4_1,
    AIModel.GPT_4_1_MINI,
    AIModel.O3,
    AIModel.O4_MINI,
}

azure_client = ChatCompletionsClient(
    endpoint=str(config.azure_endpoint),
    credential=AzureKeyCredential(config.azure_api_key.get_secret_value()),
    api_version=AZURE_API_VERSION,
    per_retry_policies=[CustomRetryPolicy()],
)


def _validate_response_content(selected_message: ChatResponseMessage) -> None:
    """
    Validate that the chat response contains content.

    Args:
        selected_message: The response message to validate

    Raises:
        HttpResponseError: If the response is empty
    """
    if not selected_message.content:
        logger.error(EMPTY_RESPONSE_ERROR_MESSAGE)
        raise HttpResponseError(EMPTY_RESPONSE_ERROR_MESSAGE)


async def _make_chat_completion_request(
    messages: list[ChatRequestMessage],
    model: AIModel,
    client: ChatCompletionsClient,
    tools: list,
) -> ChatResponseMessage:
    """
    Make a chat completion request to Azure AI.

    Args:
        messages: List of conversation messages
        model: AI model to use
        client: Azure client for making API requests
        tools: List of available tools

    Returns:
        The first choice message from the response

    Raises:
        HttpResponseError: If API request fails
        Exception: For unexpected errors
    """
    try:
        response = await client.complete(
            messages=messages,
            model=model.value,
            tools=tools,
            tool_choice=ChatCompletionsToolChoicePreset.AUTO,
        )
        return response.choices[0].message
    except HttpResponseError as e:
        logger.error(
            "HTTP response error: Status %s (%s), Message: %s",
            e.status_code,
            e.reason,
            e.message,
        )
        raise
    except Exception as error:
        logger.error("Unexpected error with model %s: %s", model.value, error)
        raise


async def _process_tool_calls_loop(
    messages: list[ChatRequestMessage],
    selected_message: ChatResponseMessage,
    client: ChatCompletionsClient,
    model: AIModel,
) -> ChatResponseMessage:
    """
    Process tool calls in a loop until no more tool calls are needed.

    Args:
        messages: The truncated messages
        selected_message: The current response message
        client: Azure client for making API requests
        model: AI model being used

    Returns:
        Final message after all tool calls are processed
    """
    while hasattr(selected_message, "tool_calls") and selected_message.tool_calls:
        selected_message = await process_tool_calls(messages, selected_message, client, model)
    return selected_message


async def complete_chat(
    messages: list[ChatRequestMessage], model: AIModel, client: ChatCompletionsClient
) -> str:
    """
    Complete a chat conversation with tool handling.

    Args:
        messages: List of conversation messages
        model: AI model to use
        client: Azure client for making API requests

    Returns:
        Text content of the final response

    Raises:
        HttpResponseError: If API request fails
        ValueError: If response is empty
    """
    message_copy = messages.copy()
    truncated_messages = truncate_messages(message_copy, model)
    tools = tool_manager.get_tool_definitions()

    selected_message = await _make_chat_completion_request(
        truncated_messages, model, client, tools
    )

    selected_message = await _process_tool_calls_loop(
        truncated_messages, selected_message, client, model
    )

    _validate_response_content(selected_message)

    return selected_message.content


def _prepare_messages_with_system(
    messages: list[ChatRequestMessage], user: User
) -> list[ChatRequestMessage]:
    """
    Prepare messages by adding system message at the beginning.

    Args:
        messages: Original chat messages
        user: User to get system message for

    Returns:
        Messages with system message prepended
    """
    system_message = get_system_message(user)
    return [system_message, *messages]


async def _handle_rate_limited_model(
    model: AIModel, messages: list[ChatRequestMessage]
) -> ModelResponse:
    """
    Handle a rate-limited model by using fallback model.

    Args:
        model: The rate-limited model
        messages: Messages to send with fallback model

    Returns:
        Tuple of (response text, fallback model used)

    Raises:
        Exception: If fallback model also fails
    """
    wait_time = rate_limit_tracker.get_wait_time(model)
    logger.warning(
        "Model %s is rate limited for %d more seconds, directly using fallback %s",
        model.value,
        wait_time,
        FALLBACK_MODEL.value,
    )
    try:
        reply = await complete_chat(messages, FALLBACK_MODEL, azure_client)
        return reply, FALLBACK_MODEL
    except Exception as error:
        logger.error("Error using fallback model %s: %s", FALLBACK_MODEL.value, error)
        raise


async def _handle_rate_limit_error(
    error: HttpResponseError, model: AIModel, messages: list[ChatRequestMessage]
) -> ModelResponse:
    """
    Handle a 429 rate limit error by setting rate limit tracker and using fallback.

    Args:
        error: The HTTP response error with status 429
        model: The model that was rate limited
        messages: Messages to send with fallback model

    Returns:
        Tuple of (response text, fallback model used)
    """
    retry_seconds = extract_retry_seconds_from_error(error.message)
    rate_limit_tracker.set_rate_limited(model, retry_seconds)

    logger.warning(
        "Rate limited on %s for %d seconds, falling back to %s",
        model.value,
        retry_seconds,
        FALLBACK_MODEL.value,
    )

    reply = await complete_chat(messages, FALLBACK_MODEL, azure_client)
    return reply, FALLBACK_MODEL


async def query_azure_chat(
    messages: list[ChatRequestMessage], user: User, model: AIModel
) -> ModelResponse:
    """
    Query Azure chat service with fallback handling.

    Args:
        messages: Chat messages to send
        user: User making the request
        model: Primary AI model to try

    Returns:
        Tuple of (response text, model actually used)
    """
    messages_with_system = _prepare_messages_with_system(messages, user)

    if rate_limit_tracker.is_rate_limited(model):
        return await _handle_rate_limited_model(model, messages_with_system)

    try:
        reply = await complete_chat(messages_with_system, model, azure_client)
        return reply, model
    except HttpResponseError as error:
        if error.status_code == RATE_LIMIT_HTTP_STATUS and model == DEFAULT_MODEL:
            return await _handle_rate_limit_error(error, model, messages_with_system)
        raise


def _ensure_image_model_support(model: AIModel) -> AIModel:
    """
    Ensure the model supports image processing, fallback to default if not.

    Args:
        model: The requested model

    Returns:
        Model that supports image processing
    """
    if model not in IMAGE_SUPPORTED_MODELS:
        return DEFAULT_MODEL
    return model


def _create_image_content_item(image_path: str) -> ImageContentItem:
    """
    Create an image content item from the given path.

    Args:
        image_path: Path to the image file

    Returns:
        ImageContentItem configured for the image
    """
    extension = Path(image_path).suffix[1:] or DEFAULT_IMAGE_EXTENSION
    return ImageContentItem(
        image_url=ImageUrl.load(
            image_file=image_path, image_format=extension, detail=ImageDetailLevel.LOW
        )
    )


def _build_image_message_content(query_text: str, image_path: str) -> list:
    """
    Build message content combining text and image.

    Args:
        query_text: Text to accompany the image
        image_path: Path to the image file

    Returns:
        List of content items for the message
    """
    message_content = []
    if query_text:
        message_content.append(TextContentItem(text=query_text))

    image_item = _create_image_content_item(image_path)
    message_content.append(image_item)
    return message_content


def _build_image_messages(
    query_text: str, image_path: str, user: User, history: ChatHistory | None
) -> list[ChatRequestMessage]:
    """
    Build the complete message array for image processing.

    Args:
        query_text: Text to accompany the image
        image_path: Path to the image file
        user: User making the request
        history: Optional chat history

    Returns:
        Complete list of messages including system, history, and current message
    """
    system_message = get_system_message(user)
    message_content = _build_image_message_content(query_text, image_path)
    user_message = UserMessage(content=message_content)

    if history:
        return [system_message, *history, user_message]
    return [system_message, user_message]


async def _handle_image_rate_limited_model(
    model: AIModel, messages: list[ChatRequestMessage]
) -> ModelResponse:
    """
    Handle a rate-limited model for image processing by using fallback model.

    Args:
        model: The rate-limited model
        messages: Messages to send with fallback model

    Returns:
        Tuple of (response text, fallback model used)

    Raises:
        Exception: If fallback model also fails
    """
    wait_time = rate_limit_tracker.get_wait_time(model)
    logger.warning(
        "Model %s is rate limited for %d more seconds, directly using fallback %s "
        "for image processing",
        model.value,
        wait_time,
        FALLBACK_MODEL.value,
    )
    try:
        reply = await complete_chat(messages, FALLBACK_MODEL, azure_client)
        return reply, FALLBACK_MODEL
    except Exception as error:
        logger.error("Error using fallback model %s for image: %s", FALLBACK_MODEL.value, error)
        raise


async def _handle_image_rate_limit_error(
    error: HttpResponseError, model: AIModel, messages: list[ChatRequestMessage]
) -> ModelResponse:
    """
    Handle a 429 rate limit error for image processing.

    Args:
        error: The HTTP response error with status 429
        model: The model that was rate limited
        messages: Messages to send with fallback model

    Returns:
        Tuple of (response text, fallback model used)
    """
    retry_seconds = extract_retry_seconds_from_error(error.message)
    rate_limit_tracker.set_rate_limited(model, retry_seconds)

    logger.warning(
        "Rate limited on %s for %d seconds, falling back to %s for image processing",
        model.value,
        retry_seconds,
        FALLBACK_MODEL.value,
    )

    reply = await complete_chat(messages, FALLBACK_MODEL, azure_client)
    return reply, FALLBACK_MODEL


async def query_azure_chat_with_image(
    image_path: str,
    query_text: str,
    user: User,
    model: AIModel,
    history: ChatHistory | None = None,
) -> ModelResponse:
    """
    Query Azure chat with image and text.

    Args:
        image_path: Path to image file
        query_text: Text to accompany the image
        user: User making the request
        model: AI model to use (must support images)
        history: Optional chat history to include in the context

    Returns:
        Tuple of (response text, model actually used)
    """
    model = _ensure_image_model_support(model)
    messages = _build_image_messages(query_text, image_path, user, history)

    if rate_limit_tracker.is_rate_limited(model):
        return await _handle_image_rate_limited_model(model, messages)

    try:
        reply = await complete_chat(messages, model, azure_client)
        return reply, model
    except HttpResponseError as error:
        if error.status_code == RATE_LIMIT_HTTP_STATUS and model == DEFAULT_MODEL:
            return await _handle_image_rate_limit_error(error, model, messages)
        raise
