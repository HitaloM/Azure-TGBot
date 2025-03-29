# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

from collections.abc import Callable

from azure.ai.inference.models import ChatCompletionsToolDefinition, FunctionDefinition

from .bing_search import bing_search
from .github_data import get_github_data

TOOL_HANDLERS: dict[str, Callable] = {
    "bing-search": bing_search,
    "get-github-data": get_github_data,
}

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
                        "description": "Recency of results.",
                        "default": "",
                    },
                    "query": {"type": "string", "description": "Search term."},
                    "user_prompt": {"type": "string", "description": "User input."},
                },
                "required": ["query", "user_prompt"],
            },
        )
    ),
    ChatCompletionsToolDefinition(
        function=FunctionDefinition(
            name="get-github-data",
            description="GET data from GitHub's REST API.",
            parameters={
                "type": "object",
                "properties": {
                    "endpoint": {
                        "type": "string",
                        "description": "GitHub REST API endpoint (include leading slash).",
                    },
                    "endpoint_description": {
                        "type": "string",
                        "description": "Short API operation description.",
                    },
                    "repo": {
                        "type": "string",
                        "description": "Repository name (owner/repo).",
                    },
                    "task": {
                        "type": "string",
                        "description": "Task description.",
                    },
                },
                "required": ["endpoint", "repo"],
            },
        )
    ),
]
