# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

import logging
from pathlib import Path

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
from bot.utils.chat.models import AIModel
from bot.utils.chat.system_message import get_system_message
from bot.utils.chat.tools.tool_manager import tool_manager

from .message_processor import truncate_messages
from .rate_limiter import rate_limit_tracker
from .retry_policy import CustomRetryPolicy
from .tool_handler import process_tool_calls
from .utils import extract_retry_seconds_from_error

logger = logging.getLogger(__name__)

ModelResponse = tuple[str, AIModel]

DEFAULT_MODEL = AIModel.GPT_4_1

IMAGE_SUPPORTED_MODELS: set[AIModel] = {
    AIModel.GPT_4_1,
    AIModel.GPT_4O_MINI,
    AIModel.O1_PREVIEW,
    AIModel.O3_MINI,
}

azure_client = ChatCompletionsClient(
    endpoint=str(config.azure_endpoint),
    credential=AzureKeyCredential(config.azure_api_key.get_secret_value()),
    api_version="2025-03-01-preview",
    per_retry_policies=[CustomRetryPolicy()],
)


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

    try:
        response = await client.complete(
            messages=truncated_messages,
            model=model.value,
            tools=tools,
            tool_choice=ChatCompletionsToolChoicePreset.AUTO,
        )
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

    selected_message: ChatResponseMessage = response.choices[0].message

    while hasattr(selected_message, "tool_calls") and selected_message.tool_calls:
        selected_message = await process_tool_calls(
            truncated_messages, selected_message, client, model
        )

    if not selected_message.content:
        msg = "Azure ChatCompletions API returned an empty response."
        logger.error(msg)
        raise HttpResponseError(msg)

    return selected_message.content


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
    system_message = get_system_message(user)
    messages_with_system = [system_message, *messages]

    if rate_limit_tracker.is_rate_limited(model):
        wait_time = rate_limit_tracker.get_wait_time(model)
        fallback_model = AIModel.GPT_4O_MINI
        logger.warning(
            "Model %s is rate limited for %d more seconds, directly using fallback %s",
            model.value,
            wait_time,
            fallback_model.value,
        )
        try:
            reply = await complete_chat(messages_with_system, fallback_model, azure_client)
            return reply, fallback_model
        except Exception as error:
            logger.error("Error using fallback model %s: %s", fallback_model.value, error)
            raise

    try:
        reply = await complete_chat(messages_with_system, model, azure_client)
        return reply, model
    except HttpResponseError as error:
        if error.status_code == 429 and model == DEFAULT_MODEL:
            retry_seconds = extract_retry_seconds_from_error(error.message)

            rate_limit_tracker.set_rate_limited(model, retry_seconds)

            fallback_model = AIModel.GPT_4O_MINI
            logger.warning(
                "Rate limited on %s for %d seconds, falling back to %s",
                model.value,
                retry_seconds,
                fallback_model.value,
            )

            reply = await complete_chat(messages_with_system, fallback_model, azure_client)
            return reply, fallback_model
        raise


async def query_azure_chat_with_image(
    image_path: str, query_text: str, user: User, model: AIModel
) -> ModelResponse:
    """
    Query Azure chat with image and text.

    Args:
        image_path: Path to image file
        query_text: Text to accompany the image
        user: User making the request
        model: AI model to use (must support images)

    Returns:
        Tuple of (response text, model actually used)
    """
    if model not in IMAGE_SUPPORTED_MODELS:
        model = DEFAULT_MODEL

    if rate_limit_tracker.is_rate_limited(model):
        wait_time = rate_limit_tracker.get_wait_time(model)
        fallback_model = AIModel.GPT_4O_MINI
        logger.warning(
            "Model %s is rate limited for %d more seconds, directly using fallback %s "
            "for image processing",
            model.value,
            wait_time,
            fallback_model.value,
        )

        system_message = get_system_message(user)
        message_content = []
        if query_text:
            message_content.append(TextContentItem(text=query_text))

        extension = Path(image_path).suffix[1:] or "jpg"
        image_item = ImageContentItem(
            image_url=ImageUrl.load(
                image_file=image_path, image_format=extension, detail=ImageDetailLevel.LOW
            )
        )
        message_content.append(image_item)

        user_message = UserMessage(content=message_content)
        messages = [system_message, user_message]

        try:
            reply = await complete_chat(messages, fallback_model, azure_client)
            return reply, fallback_model
        except Exception as error:
            logger.error(
                "Error using fallback model %s for image: %s", fallback_model.value, error
            )
            raise

    system_message = get_system_message(user)

    message_content = []
    if query_text:
        message_content.append(TextContentItem(text=query_text))

    extension = Path(image_path).suffix[1:] or "jpg"
    image_item = ImageContentItem(
        image_url=ImageUrl.load(
            image_file=image_path, image_format=extension, detail=ImageDetailLevel.LOW
        )
    )
    message_content.append(image_item)

    user_message = UserMessage(content=message_content)
    messages = [system_message, user_message]

    try:
        reply = await complete_chat(messages, model, azure_client)
        return reply, model
    except HttpResponseError as error:
        if error.status_code == 429 and model == DEFAULT_MODEL:
            retry_seconds = extract_retry_seconds_from_error(error.message)

            rate_limit_tracker.set_rate_limited(model, retry_seconds)

            fallback_model = AIModel.GPT_4O_MINI
            logger.warning(
                "Rate limited on %s for %d seconds, falling back to %s for image processing",
                model.value,
                retry_seconds,
                fallback_model.value,
            )

            reply = await complete_chat(messages, fallback_model, azure_client)
            return reply, fallback_model
        raise
