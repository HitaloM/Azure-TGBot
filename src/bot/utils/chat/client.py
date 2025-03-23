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


class ChatAPIError(Exception):
    """
    Exception raised for errors encountered during chat API requests.
    """

    pass


class ChatRateLimitError(ChatAPIError):
    """
    Exception raised for errors encountered during chat API requests.
    """

    pass


async def _complete_chat(messages: list[ChatRequestMessage], model: AIModel) -> str:
    """
    Send a chat completion request to the Azure ChatCompletions API.

    Args:
        messages (list[ChatRequestMessage]): A list of message objects to be sent.
        model (AIModel): The AI model to be used for the request.

    Returns:
        str: The content of the chat completion response.

    Raises:
        ChatAPIError: If the API returns an error or an invalid response.
    """
    try:
        async with ChatCompletionsClient(
            endpoint=str(config.azure_endpoint),
            credential=AzureKeyCredential(config.azure_api_key.get_secret_value()),
            api_version="2024-12-01-preview",
        ) as client:
            response = await client.complete(messages=messages, model=model.value)
    except HttpResponseError as e:
        error_details = e.error if hasattr(e, "error") and e.error else {}
        clean_message = getattr(error_details, "message", str(e))
        logger.error(
            "HTTP response error during API request. Status code: %s, Error message: %s",
            e.status_code if hasattr(e, "status_code") else "Unknown",
            clean_message,
        )
        raise ChatAPIError(clean_message) from e
    except Exception as error:
        logger.error("Unexpected error during API request: %s", error)
        msg = f"Unexpected API error: {error}"
        raise ChatAPIError(msg) from error

    if not response.choices or not response.choices[0].message.content:
        msg = "Azure ChatCompletions API returned an empty or invalid response."
        logger.error(msg)
        raise ChatAPIError(msg)

    return response.choices[0].message.content


async def _try_complete(messages: list[ChatRequestMessage], fallback_models: list[AIModel]) -> str:
    """
    Attempts to complete a chat request using a list of fallback AI models.

    This function iterates through the provided fallback models and tries to
    complete the chat request using each model. If a model encounters a rate
    limit error (HTTP 429), it logs a warning and continues to the next model.
    If all models fail, an exception is raised.

    Args:
        messages (list[ChatRequestMessage]): A list of chat request messages to be processed.
        fallback_models (list[AIModel]): A list of AI models to use as fallbacks for the chat
            completion.

    Returns:
        str: The completed chat response.

    Raises:
        ChatRateLimitError: If the GPT-4O-Mini model specifically encounters a rate limit error.
        ChatAPIError: If all models fail to complete the chat request or if another API
            error occurs.
    """
    for model in fallback_models:
        try:
            return await _complete_chat(messages, model)
        except ChatAPIError as e:
            if "429" in str(e):
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
    Query the Azure ChatCompletions API with a text-based message.

    Args:
        messages (list[ChatRequestMessage]): A list of message objects to send.
        user (User): The user initiating the chat request.
        model (AIModel): The AI model to be used for the request.

    Returns:
        str: The content of the chat completion response.
    """
    system_message = get_system_message(user)
    messages_with_system = [system_message, *messages]
    fallback_models = [model, AIModel.GPT_4O_MINI] if model != AIModel.GPT_4O_MINI else [model]
    return await _try_complete(messages_with_system, fallback_models)


async def query_azure_chat_with_image(
    image_path: str, query_text: str, user: User, model: AIModel
) -> str:
    """
    Query the Azure ChatCompletions API with an image and accompanying text.

    Args:
        image_path (str): The file path of the image to send.
        query_text (str): The text query to accompany the image.
        user (User): The user initiating the chat request.
        model (AIModel): The AI model to be used for the request.

    Returns:
        str: The content of the chat completion response.
    """
    if model not in {  # Only OpenAI models support image-based completions.
        AIModel.GPT_4O,
        AIModel.GPT_4O_MINI,
        AIModel.O1,
        AIModel.O1_MINI,
        AIModel.O3_MINI,
    }:
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
    fallback_models = [model, AIModel.GPT_4O_MINI] if model != AIModel.GPT_4O_MINI else [model]
    return await _try_complete(messages, fallback_models)
