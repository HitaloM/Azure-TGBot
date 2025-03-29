# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

from collections.abc import Callable

from azure.ai.inference.models import ChatCompletionsToolDefinition, FunctionDefinition

from .bing_search import bing_search

TOOL_HANDLERS: dict[str, Callable] = {"bing-search": bing_search}

TOOLS: list[ChatCompletionsToolDefinition] = [
    ChatCompletionsToolDefinition(
        function=FunctionDefinition(
            name="bing-search",
            description="Search the web using Bing.",
            parameters={
                "type": "object",
                "properties": {
                    "freshness": {
                        "type": "string",
                        "description": "How recent the results are.",
                        "default": "",
                    },
                    "query": {"type": "string", "description": "Search term."},
                    "user_prompt": {"type": "string", "description": "User's input."},
                },
                "required": ["query", "user_prompt"],
            },
        )
    ),
]
