# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

import logging
import time
from collections.abc import Callable
from inspect import iscoroutinefunction
from pathlib import Path

import orjson
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
from .tools.bing_search import bing_search
from .tools.open_web_results import open_url, open_web_results
from .tools.schema import TOOLS

logger = logging.getLogger(__name__)

DEFAULT_MODEL = AIModel.GPT_4O


class CustomRetryPolicy(RetryPolicy):
    """
    CustomRetryPolicy implements a specialized retry mechanism for HTTP requests by extending
    the base RetryPolicy. It customizes the retry loop by adjusting timeouts, handling HTTP 429
    ("Too Many Requests") scenarios, and managing specific exceptions such as
    ClientAuthenticationError and other Azure-related errors.

    Methods:
        send(request: PipelineRequest) -> PipelineResponse:
            Executes the provided HTTP request within a retry loop.
            It performs the following tasks:

            - Configures retry settings based on the context options.
            - Adjusts the timeout settings and tracks retry attempts.
            - Sends the request using an underlying transport and evaluates the response.
            - Checks if the response requires a retry (e.g., on encountering HTTP 429) and, if so,
              increments the retry count and delays the next attempt.
            - Handles specific exceptions by determining whether a retry is applicable based on
              the error type.
            - Updates the request context after a successful attempt or raises an error once the
              maximum retries are exceeded.

            This method ensures that transient failures and rate-limiting issues are managed
            gracefully while providing a mechanism for adhering to an absolute timeout
            constraint across multiple retry attempts.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def send(self, request: PipelineRequest) -> PipelineResponse:
        """
        Sends a pipeline request and returns the corresponding response,
        applying retry logic based on the provided options.

        This asynchronous method configures retry settings, adjusts the
        request timeout, and handles retries in case of errors such as HTTP 429
        (Too Many Requests) or transient Azure errors. It updates retry history,
        computes elapsed time for the absolute timeout, and optionally sleeps
        between retries if necessary. The method ultimately returns a
        PipelineResponse if successful, or raises an exception if the maximum
        retries are exceeded or if a non-retryable error occurs.

        Parameters:
            request (PipelineRequest): The request object containing the HTTP request, context
                (including options and transport), and other related data.

        Returns:
            PipelineResponse: The response object received after sending the
                request.

        Raises:
            HttpResponseError: When the HTTP response status code is 429 indicating too many
                requests.
            ClientAuthenticationError: If authentication fails.
            AzureError: For errors encountered during the sending process or if the maximum
                retry timeout is exceeded.
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
                    print(response.http_response)
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

            except ClientAuthenticationError:
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
                raise

            finally:
                elapsed = time.time() - start_time
                if absolute_timeout:
                    absolute_timeout -= elapsed

        if response is None:
            msg = "Maximum retries exceeded."
            raise AzureError(msg)

        self.update_context(response.context, retry_settings)
        return response


def get_azure_client() -> ChatCompletionsClient:
    """
    Creates and returns an instance of the ChatCompletionsClient configured for Azure OpenAI.

    This function initializes the client with the specified endpoint, API key, API version,
    and a custom retry policy.

    Returns:
        ChatCompletionsClient: An instance of the Azure OpenAI ChatCompletionsClient.

    Raises:
        ValueError: If the configuration values (e.g., endpoint or API key) are invalid.
    """
    return ChatCompletionsClient(
        endpoint=str(config.azure_endpoint),
        credential=AzureKeyCredential(config.azure_api_key.get_secret_value()),
        api_version="2025-03-01-preview",
        per_retry_policies=[CustomRetryPolicy()],
    )


IMAGE_SUPPORTED_MODELS: set[AIModel] = {
    AIModel.GPT_4O,
    AIModel.GPT_4O_MINI,
    AIModel.O1,
    AIModel.O1_MINI,
    AIModel.O3_MINI,
}

TOOL_HANDLERS: dict[str, Callable] = {
    "bing-search": bing_search,
    "open-web-results": open_web_results,
    "open-url": open_url,
}


async def _execute_tool_call(function_name: str, arguments: dict) -> str:
    """
    Executes a tool call by invoking the appropriate handler based on the function name.

    Args:
        function_name (str): The name of the function to call.
        arguments (dict): A dictionary of arguments for the function call.

    Returns:
        str: The JSON-encoded result of the function execution.
    """
    handler = TOOL_HANDLERS.get(function_name)
    if not handler:
        return orjson.dumps({"error": f"Unknown function {function_name}"}).decode()

    result = await handler(**arguments) if iscoroutinefunction(handler) else handler(**arguments)
    return orjson.dumps(result).decode()


async def _process_tool_calls(
    messages: list[ChatRequestMessage],
    selected_message: ChatResponseMessage,
    client: ChatCompletionsClient,
    model_value: str,
) -> ChatResponseMessage:
    """
    Processes tool calls in the response message and updates the messages list accordingly.

    Args:
        messages (list[ChatRequestMessage]): List of chat messages.
        selected_message (ChatResponseMessage): The response message containing tool_calls.
        client (ChatCompletionsClient): Client used to communicate with the Azure service.
        model_value (str): The model identifier used for the chat request.

    Returns:
        ChatResponseMessage: An updated chat response message after processing tool calls.
    """
    for tool_call in selected_message.tool_calls or []:
        function_name = tool_call.function.name
        try:
            function_args = orjson.loads(tool_call.function.arguments)
        except Exception as error:
            logger.error(
                "Falha ao decodificar argumentos da ferramenta '%s': %s", function_name, error
            )
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
        try:
            tool_response = await _execute_tool_call(function_name, function_args)
        except Exception as error:
            logger.error(
                "Erro na execução da chamada da ferramenta '%s': %s", function_name, error
            )
            tool_response = orjson.dumps({
                "error": f"Erro na execução de {function_name}"
            }).decode()
        messages.append(ToolMessage(content=tool_response, tool_call_id=tool_call.id))
    return (await client.complete(messages=messages, model=model_value)).choices[0].message


async def _complete_chat(
    messages: list[ChatRequestMessage], model: AIModel, client: ChatCompletionsClient
) -> str:
    """
    Asynchronously completes a chat conversation using the Azure ChatCompletions API.

    Sends the list of chat messages to the API and processes any tool calls that may occur,
    returning the content of the final response message.

    Args:
        messages (list[ChatRequestMessage]): List of messages to be sent.
        model (AIModel): The AI model to be utilized.
        client (ChatCompletionsClient): Client instance for making the API request.

    Returns:
        str: The content of the response message obtained.

    Raises:
        HttpResponseError: If an error occurs during the request or the response is invalid.
    """
    try:
        response = await client.complete(
            messages=messages,
            model=model.value,
            tools=TOOLS,
            tool_choice=ChatCompletionsToolChoicePreset.AUTO,
        )
    except HttpResponseError as e:
        logger.error(
            "HTTP response error during API request. Status code: %s (%s), Error message: %s",
            e.status_code,
            e.reason,
            e.message,
        )
        raise
    except Exception as error:
        logger.error("Unexpected error during API request with model %s: %s", model.value, error)
        raise

    selected_message: ChatResponseMessage = response.choices[0].message

    while hasattr(selected_message, "tool_calls") and selected_message.tool_calls:
        selected_message = await _process_tool_calls(
            messages, selected_message, client, model.value
        )

    if not selected_message.content:
        msg = "Azure ChatCompletions API returned an empty or invalid response."
        logger.error(msg)
        raise HttpResponseError(msg)

    return selected_message.content


async def query_azure_chat(
    messages: list[ChatRequestMessage], user: User, model: AIModel
) -> tuple[str, AIModel]:
    """
    Queries the Azure chat service using a list of messages and returns the response.

    A system message is generated from the user and prepended to the provided messages before
    sending the request via the Azure ChatCompletions API.

    Args:
        messages (list[ChatRequestMessage]): List of messages for the query.
        user (User): The user initiating the query.
        model (AIModel): The AI model to be used for the conversation.

    Returns:
        tuple[str, AIModel]: The response from the Azure chat service and the model used.
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
            async with get_azure_client() as client:
                reply = await _complete_chat(messages_with_system, fallback_model, client)
                return reply, fallback_model
        raise


async def query_azure_chat_with_image(
    image_path: str, query_text: str, user: User, model: AIModel
) -> tuple[str, AIModel]:
    """
    Queries an Azure chat model with a combination of image and text.

    If the provided model does not support images, a default model is used.
    The image is read from the given path, processed, and sent along with a text query.

    Args:
        image_path (str): The file path to the image.
        query_text (str): The text query to accompany the image.
        user (User): The user initiating the query.
        model (AIModel): The AI model to be used for the query.

    Returns:
        tuple[str, AIModel]: The response from the Azure chat service and the model used.
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
    try:
        async with get_azure_client() as client:
            reply = await _complete_chat(messages, model, client)
            return reply, model
    except HttpResponseError as error:
        if error.status_code == 429 and model == AIModel.GPT_4O:
            fallback_model = AIModel.GPT_4O_MINI
            async with get_azure_client() as client:
                reply = await _complete_chat(messages, fallback_model, client)
                return reply, fallback_model
        raise
