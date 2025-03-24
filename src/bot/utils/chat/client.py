# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

import logging
from pathlib import Path

from aiogram.types import User
from azure.ai.inference.aio import ChatCompletionsClient
from azure.ai.inference.models import (
    ChatRequestMessage,
    ImageContentItem,
    ImageDetailLevel,
    ImageUrl,
    TextContentItem,
    UserMessage,
)
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

from bot import config

from .models import DEFAULT_MODEL, AIModel
from .system_message import get_system_message

logger = logging.getLogger(__name__)

IMAGE_SUPPORTED_MODELS: set[AIModel] = {
    AIModel.GPT_4O,
    AIModel.GPT_4O_MINI,
    AIModel.O1,
    AIModel.O1_MINI,
    AIModel.O3_MINI,
}


class ChatAPIError(Exception):
    """
    Exception raised for errors encountered while interacting with the Chat API.

    Attributes:
        message (str): A description of the error.
        status_code (int | None): The HTTP status code associated with the error, if available.
        reason (str | None): Additional information or reason for the error, if provided.
    """

    def __init__(
        self, message: str, status_code: int | None = None, reason: str | None = None
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.reason = reason


class ChatRateLimitError(ChatAPIError):
    """
    Exception raised when the chat API rate limit is exceeded.

    This error is intended to signal that the client has made too many
    requests to the chat API within a given time frame, and further
    requests are temporarily blocked until the rate limit resets.

    Attributes:
        Inherits all attributes from the base `ChatAPIError` class.
    """

    pass


async def _complete_chat(messages: list[ChatRequestMessage], model: AIModel) -> str:
    """
    Asynchronously completes a chat conversation using the Azure ChatCompletions API.

    This function sends a list of chat messages to the Azure ChatCompletions API and retrieves
    a response based on the specified AI model. It handles API errors and ensures the response
    is valid before returning the content of the first message choice.

    Args:
        messages (list[ChatRequestMessage]): A list of chat messages to send to the API.
        model (AIModel): The AI model to use for generating the chat completion.

    Returns:
        str: The content of the first message choice returned by the API.

    Raises:
        ChatAPIError: If an error occurs during the API request or if the response is invalid.
    """
    try:
        async with ChatCompletionsClient(
            endpoint=str(config.azure_endpoint),
            credential=AzureKeyCredential(config.azure_api_key.get_secret_value()),
            api_version="2024-12-01-preview",
        ) as client:
            response = await client.complete(messages=messages, model=model.value)
    except HttpResponseError as e:
        status_code = getattr(e, "status_code", None)
        reason = getattr(e, "reason", None)
        message = getattr(e, "message", str(e))
        logger.error(
            "HTTP response error during API request. Status code: %s (%s), Error message: %s",
            status_code,
            reason,
            message,
        )
        raise ChatAPIError(message, status_code, reason) from e
    except Exception as error:
        logger.error("Unexpected error during API request with model %s: %s", model.value, error)
        msg = f"Unexpected API error when using model {model.value}: {error}"
        raise ChatAPIError(msg) from error

    if not response.choices or not response.choices[0].message.content:
        msg = "Azure ChatCompletions API returned an empty or invalid response."
        logger.error(msg)
        raise ChatAPIError(msg)

    return response.choices[0].message.content


async def _try_complete(messages: list[ChatRequestMessage], fallback_models: list[AIModel]) -> str:
    """
    Attempts to complete a chat request using a list of fallback AI models.

    This function iterates through the provided fallback models and tries to complete
    the chat request using each model. If a model encounters a rate limit error (HTTP 429),
    it logs a warning and continues to the next model. If all models fail, an exception
    is raised.

    Args:
        messages (list[ChatRequestMessage]): A list of chat request messages to process.
        fallback_models (list[AIModel]): A list of AI models to use as fallbacks for the chat
            completion.

    Returns:
        str: The completed chat response.

    Raises:
        ChatAPIError: If all models fail to complete the chat request or if an error other than
            rate limiting occurs.
        ChatRateLimitError: If the GPT-4O-Mini model encounters a rate limit error.
    """
    for model in fallback_models:
        try:
            return await _complete_chat(messages, model)
        except ChatAPIError as e:
            if e.status_code == 429:
                logger.warning("Rate limit reached for model %s.", model.value)
                if model == AIModel.GPT_4O_MINI:
                    msg = "Rate limit reached for GPT-4O-Mini. Please try again later."
                    raise ChatRateLimitError(msg) from e
                continue
            raise
    msg = "All models failed."
    raise ChatAPIError(msg)


async def query_azure_chat(messages: list[ChatRequestMessage], user: User, model: AIModel) -> str:
    """
    Queries the Azure chat service with a list of messages and returns the response.

    Args:
        messages (list[ChatRequestMessage]): A list of chat messages to send to the Azure chat
            service.
        user (User): The user initiating the chat query.
        model (AIModel): The AI model to use for the chat query. If the model is
            `AIModel.GPT_4O_MINI`, it will be used directly; otherwise, a fallback to
            `AIModel.GPT_4O_MINI` will be attempted.

    Returns:
        str: The response from the Azure chat service.

    Raises:
        Any exceptions raised by the `_try_complete` function.
    """
    system_message = get_system_message(user)
    messages_with_system = [system_message, *messages]
    fallback_models = [model] if model == AIModel.GPT_4O_MINI else [model, AIModel.GPT_4O_MINI]
    return await _try_complete(messages_with_system, fallback_models)


async def query_azure_chat_with_image(
    image_path: str, query_text: str, user: User, model: AIModel
) -> str:
    """
    Queries an Azure chat model with a combination of text and an image.

    Args:
        image_path (str): The file path to the image to be sent with the query.
        query_text (str): The text query to be sent to the chat model.
        user (User): The user initiating the query, used to generate a system message.
        model (AIModel): The AI model to be used for the query. If the model does not
            support images, a default model will be used.

    Returns:
        str: The response from the Azure chat model.

    Notes:
        - If the provided model does not support image inputs, the function will
          automatically fall back to a default model.
        - The function attempts to use a fallback model if the primary model fails.
    """
    if model not in IMAGE_SUPPORTED_MODELS:
        model = DEFAULT_MODEL

    system_message = get_system_message(user)
    message_text = TextContentItem(text=query_text)
    extension = Path(image_path).suffix[1:] or "jpg"
    image_item = ImageContentItem(
        image_url=ImageUrl.load(
            image_file=image_path, image_format=extension, detail=ImageDetailLevel.LOW
        )
    )
    user_message = UserMessage(content=[message_text, image_item])
    messages = [system_message, user_message]
    fallback_models = [model] if model == AIModel.GPT_4O_MINI else [model, AIModel.GPT_4O_MINI]
    return await _try_complete(messages, fallback_models)
