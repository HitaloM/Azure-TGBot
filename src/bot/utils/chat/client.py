# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

import logging
import time
from pathlib import Path
from typing import Any

import orjson
import tiktoken
from aiogram.types import User
from azure.ai.inference.aio import ChatCompletionsClient
from azure.ai.inference.models import (
    AssistantMessage,
    ChatCompletionsToolCall,
    ChatCompletionsToolChoicePreset,
    ChatRequestMessage,
    ChatResponseMessage,
    FunctionCall,
    ImageContentItem,
    ImageDetailLevel,
    ImageUrl,
    TextContentItem,
    ToolMessage,
    UserMessage,
)
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import (
    AzureError,
    ClientAuthenticationError,
    HttpResponseError,
    ServiceRequestError,
)
from azure.core.pipeline import (
    PipelineRequest,
    PipelineResponse,
)
from azure.core.pipeline.policies import RetryPolicy

from bot import config

from .models import AIModel
from .system_message import get_system_message
from .tools.tool_manager import tool_manager

logger = logging.getLogger(__name__)

type ToolResult = dict[str, Any]
type ModelResponse = tuple[str, AIModel]

DEFAULT_MODEL = AIModel.GPT_4O

IMAGE_SUPPORTED_MODELS: set[AIModel] = {
    AIModel.GPT_4O,
    AIModel.GPT_4O_MINI,
    AIModel.O1_PREVIEW,
    AIModel.O3_MINI,
}

CHAT_PARAMS = {
    "temperature": 0.8,
    "top_p": 0.9,
    "max_tokens": 1000,
}


class CustomRetryPolicy(RetryPolicy):
    """
    Custom retry policy for handling rate limits and transient errors.

    Extends the RetryPolicy to provide specialized handling for HTTP 429 errors
    (Too Many Requests) and other transient Azure service errors.
    """

    async def send(self, request: PipelineRequest) -> PipelineResponse:
        """
        Send request with retry handling for API rate limits and failures.

        Args:
            request: Pipeline request to send

        Returns:
            Response from the pipeline

        Raises:
            ClientAuthenticationError: When authentication fails
            AzureError: For other errors after exceeding retries
        """
        retry_settings = self.configure_retries(request.context.options)
        self._configure_positions(request, retry_settings)
        absolute_timeout = retry_settings["timeout"]
        is_response_error = True
        response = None

        while True:
            start_time = time.time()
            transport = request.context.transport

            try:
                self._configure_timeout(request, absolute_timeout, is_response_error)
                request.context["retry_count"] = len(retry_settings["history"])
                response = await self.next.send(request)  # type: ignore

                if response.http_response.status_code == 429:
                    logger.warning("HTTP 429 Too Many Requests encountered.")
                    raise HttpResponseError(
                        message="Too many requests", response=response.http_response
                    )

                if self.is_retry(retry_settings, response) and self.increment(
                    retry_settings, response=response
                ):
                    await self.sleep(retry_settings, transport, response=response)  # type: ignore
                    is_response_error = True
                    continue

                break

            except ClientAuthenticationError as e:
                logger.error("Client authentication failed: %s", e)
                raise

            except AzureError as err:
                if (
                    absolute_timeout > 0
                    and self._is_method_retryable(retry_settings, request.http_request)
                    and self.increment(retry_settings, response=request, error=err)
                ):
                    await self.sleep(retry_settings, transport)  # type: ignore
                    is_response_error = not isinstance(err, ServiceRequestError)
                    continue
                logger.error("Azure error encountered: %s", err)
                raise

            finally:
                elapsed = time.time() - start_time
                if absolute_timeout:
                    absolute_timeout -= elapsed

        if response is None:
            msg = "Maximum retries exceeded."
            logger.error(msg)
            raise AzureError(msg)

        self.update_context(response.context, retry_settings)
        return response


def get_azure_client() -> ChatCompletionsClient:
    """
    Create a configured Azure ChatCompletions client.

    Returns:
        Configured ChatCompletionsClient with custom retry policy
    """
    client = ChatCompletionsClient(
        endpoint=str(config.azure_endpoint),
        credential=AzureKeyCredential(config.azure_api_key.get_secret_value()),
        api_version="2025-03-01-preview",
        per_retry_policies=[CustomRetryPolicy()],
    )
    logger.info("Azure ChatCompletionsClient successfully created.")
    return client


async def _execute_tool_call(function_name: str, arguments: dict[str, Any]) -> str:
    """
    Execute a tool call with error handling.

    Args:
        function_name: Name of function to call
        arguments: Arguments to pass to function

    Returns:
        JSON-encoded result or error message
    """
    handler = tool_manager.get_tool_handlers().get(function_name)
    if not handler:
        return orjson.dumps({"error": f"Unknown function {function_name}"}).decode()

    try:
        result = await handler(**arguments)
        return orjson.dumps(result).decode()
    except Exception as error:
        logger.error("Error executing tool call '%s': %s", function_name, error)
        return orjson.dumps({"error": f"Error executing {function_name}: {error!s}"}).decode()


async def _process_tool_calls(
    messages: list[ChatRequestMessage],
    message_with_tools: ChatResponseMessage,
    client: ChatCompletionsClient,
    model: AIModel,
) -> ChatResponseMessage:
    """
    Process tool calls from a response message and get AI's follow-up response.

    Args:
        messages: Existing conversation messages
        message_with_tools: Response message containing tool calls
        client: Azure client for making follow-up requests
        model: AI model being used

    Returns:
        Updated chat response after tool execution
    """
    for tool_call in message_with_tools.tool_calls or []:
        function_name = tool_call.function.name
        try:
            function_args = orjson.loads(tool_call.function.arguments)
        except Exception as error:
            logger.error("Failed to decode tool arguments for '%s': %s", function_name, error)
            function_args = {}

        messages.append(
            AssistantMessage(
                tool_calls=[
                    ChatCompletionsToolCall(
                        id=tool_call.id,
                        function=FunctionCall(
                            name=function_name,
                            arguments=tool_call.function.arguments,
                        ),
                    )
                ]
            )
        )

        tool_response = await _execute_tool_call(function_name, function_args)
        messages.append(ToolMessage(content=tool_response, tool_call_id=tool_call.id))

    if model not in {AIModel.DEEPSEEK_V3, AIModel.DEEPSEEK_R1} and (
        (last_message := messages[-1])
        and isinstance(last_message, ToolMessage)
        and isinstance(last_message.content, str)
    ):
        enc = tiktoken.encoding_for_model(model.value)
        tokens = enc.encode(last_message.content)
        if len(tokens) > config.token_truncate_limit:
            last_message.content = enc.decode(tokens[: config.token_truncate_limit])

    return (
        (await client.complete(messages=messages, model=model.value, **CHAT_PARAMS))
        .choices[0]
        .message
    )


async def _complete_chat(
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
    tools = tool_manager.get_tool_definitions()

    try:
        response = await client.complete(
            messages=messages,
            model=model.value,
            tools=tools,
            tool_choice=ChatCompletionsToolChoicePreset.AUTO,
            **CHAT_PARAMS,
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
        selected_message = await _process_tool_calls(messages, selected_message, client, model)

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

    try:
        async with get_azure_client() as client:
            reply = await _complete_chat(messages_with_system, model, client)
            return reply, model
    except HttpResponseError as error:
        if error.status_code == 429 and model == AIModel.GPT_4O:
            fallback_model = AIModel.GPT_4O_MINI
            logger.warning(
                "Rate limited on %s, falling back to %s", model.value, fallback_model.value
            )
            async with get_azure_client() as client:
                reply = await _complete_chat(messages_with_system, fallback_model, client)
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
        async with get_azure_client() as client:
            reply = await _complete_chat(messages, model, client)
            return reply, model
    except HttpResponseError as error:
        if error.status_code == 429 and model == AIModel.GPT_4O:
            fallback_model = AIModel.GPT_4O_MINI
            logger.warning(
                "Rate limited on %s, falling back to %s", model.value, fallback_model.value
            )
            async with get_azure_client() as client:
                reply = await _complete_chat(messages, fallback_model, client)
                return reply, fallback_model
        raise
