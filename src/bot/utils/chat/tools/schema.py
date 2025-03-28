# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

from azure.ai.inference.models import ChatCompletionsToolDefinition, FunctionDefinition

TOOLS: list[ChatCompletionsToolDefinition] = [
    ChatCompletionsToolDefinition(
        function=FunctionDefinition(
            name="bing-search",
            description=(
                "Performs a Bing web search based on the user's query. "
                "Rewrites and optimizes the query and supports a freshness filter."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "freshness": {
                        "type": "string",
                        "description": (
                            "Freshness refers to the date that Bing originally discovered the "
                            "website, not when the publisher published the website. Valid values: "
                            "'' - return websites that Bing discovered at any time, 'month' "
                            "- return websites that Bing discovered within the last 30 days, "
                            "'week' - return websites that Bing discovered within the last 7 "
                            "days, 'day' - return websites discovered by Bing within the last "
                            "24 hours, or a specific date range in the form "
                            "'YYYY-MM-DD..YYYY-MM-DD'."
                        ),
                        "default": "",
                    },
                    "query": {
                        "type": "string",
                        "description": (
                            "A query string based on the user's request. Rewritten and "
                            "optimized for an effective Bing web search."
                        ),
                    },
                },
                "required": ["query"],
            },
        )
    ),
    ChatCompletionsToolDefinition(
        function=FunctionDefinition(
            name="open-web-results",
            description=(
                "Extracts and returns the web search results from the Bing search response."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "search_results": {
                        "type": "array",
                        "description": "The Bing search results.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "url": {"type": "string", "format": "uri"},
                                "snippet": {"type": "string"},
                            },
                            "required": ["title", "url", "snippet"],
                        },
                        "required": ["user_prompt", "results"],
                    }
                },
                "required": ["search_results"],
            },
        )
    ),
    ChatCompletionsToolDefinition(
        function=FunctionDefinition(
            name="open-url",
            description="Opens a specified URL to retrieve its content.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "format": "uri",
                        "description": "The URL to open and retrieve content from.",
                    }
                },
                "required": ["url"],
            },
        )
    ),
]
