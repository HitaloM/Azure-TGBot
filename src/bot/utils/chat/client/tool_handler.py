# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

import logging
from typing import Any

import orjson
from azure.ai.inference.aio import ChatCompletionsClient
from azure.ai.inference.models import (
    AssistantMessage,
    ChatCompletionsToolCall,
    ChatRequestMessage,
    ChatResponseMessage,
    FunctionCall,
    ToolMessage,
)

from bot import config
from bot.utils.chat.models import AIModel
from bot.utils.chat.tools.tool_manager import tool_manager

from .message_processor import get_token_encoding, truncate_messages

logger = logging.getLogger(__name__)

ToolResult = dict[str, Any]


async def execute_tool_call(function_name: str, arguments: dict[str, Any]) -> str:
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


async def process_tool_calls(
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

        tool_response = await execute_tool_call(function_name, function_args)

        enc = get_token_encoding(model)
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

    truncated_messages = truncate_messages(messages, model)

    messages.clear()
    messages.extend(truncated_messages)

    return (await client.complete(messages=messages, model=model.value)).choices[0].message
