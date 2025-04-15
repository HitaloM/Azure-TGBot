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

DEFAULT_MODEL = AIModel.GPT_4_1

IMAGE_SUPPORTED_MODELS: set[AIModel] = {
    AIModel.GPT_4_1,
    AIModel.GPT_4O_MINI,
    AIModel.O1_PREVIEW,
    AIModel.O3_MINI,
}

INPUT_TOKEN_LIMIT = config.token_truncate_limit


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


azure_client = ChatCompletionsClient(
    endpoint=str(config.azure_endpoint),
    credential=AzureKeyCredential(config.azure_api_key.get_secret_value()),
    api_version="2025-03-01-preview",
    per_retry_policies=[CustomRetryPolicy()],
)


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

        # Truncate tool response if too large
        encoding_name = model.value if model != AIModel.GPT_4_1 else "gpt-4o"
        try:
            enc = tiktoken.encoding_for_model(encoding_name)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")

        tokens = enc.encode(tool_response)
        if len(tokens) > config.token_truncate_limit // 2:
            logger.info(
                "Truncating tool response from %d to %d tokens for model %s",
                len(tokens),
                config.token_truncate_limit // 2,
                model.value,
            )
            tool_response = enc.decode(tokens[: config.token_truncate_limit // 2]) + "..."

        messages.append(ToolMessage(content=tool_response, tool_call_id=tool_call.id))

    # After processing all tool calls, truncate the entire conversation if needed
    truncated_messages = _truncate_messages(messages, model)

    # Replace the original messages with the truncated ones to maintain the limit
    messages.clear()
    messages.extend(truncated_messages)

    return (await client.complete(messages=messages, model=model.value)).choices[0].message


def _truncate_content_by_tokens(content: str, model: AIModel, max_tokens: int) -> str:
    """
    Truncate content to a maximum token count.

    Args:
        content: Text content to truncate
        model: AI model to determine token encoding
        max_tokens: Maximum number of tokens to keep

    Returns:
        Truncated content that fits within token limit
    """
    encoding_name = model.value if model != AIModel.GPT_4_1 else "gpt-4o"
    try:
        enc = tiktoken.encoding_for_model(encoding_name)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")  # Default fallback encoding

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


def _get_message_token_count(message: ChatRequestMessage, model: AIModel) -> int:
    """
    Calculate token count for a message.

    Args:
        message: The message to calculate tokens for
        model: AI model to determine token encoding

    Returns:
        Approximate token count
    """
    encoding_name = model.value if model != AIModel.GPT_4_1 else "gpt-4o"
    try:
        enc = tiktoken.encoding_for_model(encoding_name)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")

    # Extract text content based on message type
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


def _truncate_messages(
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
    # Make a copy to avoid modifying the original
    messages_copy = list(messages)

    # First message is usually system message, keep it intact
    if not messages_copy:
        return messages_copy

    system_message = messages_copy[0]

    # If only system message or very few messages, no need to truncate
    if len(messages_copy) <= 3:
        return messages_copy

    remaining_messages = messages_copy[1:]

    # Calculate system message tokens
    system_tokens = _get_message_token_count(system_message, model)
    available_tokens = INPUT_TOKEN_LIMIT - system_tokens

    # Extract and preserve newest messages
    newest_messages = _extract_newest_messages(remaining_messages)
    newest_tokens = sum(_get_message_token_count(msg, model) for msg in newest_messages)
    available_tokens -= newest_tokens

    # Process older messages within token constraints
    kept_messages = _keep_messages_within_limit(
        list(reversed(remaining_messages[: -len(newest_messages)])), available_tokens, model
    )

    # Reassemble the conversation: system message + kept messages + newest messages
    truncated_messages = [system_message, *kept_messages, *newest_messages]

    if len(truncated_messages) < len(messages):
        logger.info(
            "Truncated conversation from %d to %d messages to fit token limit",
            len(messages),
            len(truncated_messages),
        )

    return truncated_messages


def _extract_newest_messages(messages: list[ChatRequestMessage]) -> list[ChatRequestMessage]:
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

    # If there's a response to the last message, include it
    if len(messages) >= 2 and isinstance(messages[-2], AssistantMessage):
        newest.insert(0, messages[-2])

    return newest


def _keep_messages_within_limit(
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
        message_tokens = _get_message_token_count(message, model)

        if message_tokens <= available_tokens:
            # Message fits completely
            kept_messages.append(message)
            available_tokens -= message_tokens
        elif available_tokens > 0:
            # Try to include a truncated version of the message
            truncated_message = _try_truncate_message(message, available_tokens, model)
            if truncated_message:
                kept_messages.append(truncated_message)
            available_tokens = 0
            break

        if available_tokens <= 0:
            break

    # Restore chronological order
    kept_messages.reverse()
    return kept_messages


def _try_truncate_message(
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
    # Reserve tokens for ellipsis
    truncate_limit = available_tokens - 3
    if truncate_limit <= 0:
        return None

    if isinstance(message, UserMessage) and isinstance(message.content, list):
        # Handle UserMessage with content items
        for i, item in enumerate(message.content):
            if isinstance(item, TextContentItem) and item.text:
                truncated_text = _truncate_content_by_tokens(item.text, model, truncate_limit)
                if truncated_text != item.text:
                    truncated_text += "..."
                    # Create a new content list to avoid modifying the original
                    new_content = list(message.content)
                    new_content[i] = TextContentItem(text=truncated_text)
                    return UserMessage(content=new_content)

    elif isinstance(message, (UserMessage, AssistantMessage)) and isinstance(message.content, str):
        # Handle messages with string content
        truncated_content = _truncate_content_by_tokens(message.content, model, truncate_limit)
        if truncated_content != message.content:
            truncated_content += "..."
            if isinstance(message, AssistantMessage):
                return AssistantMessage(content=truncated_content)
            return UserMessage(content=truncated_content)

    # No truncation was possible
    return None


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
    # Make a copy of messages to avoid modifying the original messages
    message_copy = messages.copy()

    # Truncate messages to prevent hitting token limits
    truncated_messages = _truncate_messages(message_copy, model)

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
        selected_message = await _process_tool_calls(
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

    try:
        reply = await _complete_chat(messages_with_system, model, azure_client)
        return reply, model
    except HttpResponseError as error:
        if error.status_code == 429 and model == DEFAULT_MODEL:
            fallback_model = AIModel.GPT_4O_MINI
            logger.warning(
                "Rate limited on %s, falling back to %s", model.value, fallback_model.value
            )
            reply = await _complete_chat(messages_with_system, fallback_model, azure_client)
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
        reply = await _complete_chat(messages, model, azure_client)
        return reply, model
    except HttpResponseError as error:
        if error.status_code == 429 and model == DEFAULT_MODEL:
            fallback_model = AIModel.GPT_4O_MINI
            logger.warning(
                "Rate limited on %s, falling back to %s", model.value, fallback_model.value
            )
            reply = await _complete_chat(messages, fallback_model, azure_client)
            return reply, fallback_model
        raise
