# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

import asyncio
import logging
from typing import Any, ClassVar, TypedDict

import aiohttp
from lxml import html

from bot import config

from .base_tool import BaseTool
from .tool_manager import tool_manager

logger = logging.getLogger(__name__)


class SearchResultItem(TypedDict):
    """Type definition for a search result item."""

    title: str
    url: str
    snippet: str
    content: str


def parse_content(html_content: str, max_characters: int | None = None) -> str:
    """
    Parses the given HTML content and extracts readable text while removing unnecessary elements.

    This function removes script, style, header, footer, navigation, and aside elements from
    the HTML content, then extracts the text content from the body. It also cleans up non-ASCII
    characters and excessive whitespace.

    Args:
        html_content: The HTML content to parse.
        max_characters: The maximum number of characters to include in
            the output. If None, the entire cleaned text is returned. Defaults to None.

    Returns:
        The cleaned and extracted text content from the HTML.
    """
    try:
        tree = html.fromstring(html_content)
        for element in tree.xpath("//script | //style | //header | //footer | //nav | //aside"):
            parent = element.getparent()
            if parent is not None:
                parent.remove(element)

        text = tree.xpath("string(body)")
        text = " ".join(text.split())

        return text[:max_characters].strip() if max_characters else text.strip()
    except Exception as e:
        logger.error("[Bing Search] - Error parsing HTML content: %s", e)
        return ""


async def fetch_page(session: aiohttp.ClientSession, url: str) -> str | None:
    """
    Fetches the content of a web page asynchronously.

    Args:
        session: The aiohttp session to use for making the HTTP request.
        url: The URL of the web page to fetch.

    Returns:
        The content of the web page as a string if the request is successful,
        or None if the URL is invalid, the page is not found (404), or an error occurs during the
        request.
    """
    if not url or not url.startswith(("http://", "https://")):
        logger.warning("[Bing Search] - Invalid URL: %s", url)
        return None

    try:
        async with session.get(
            url, allow_redirects=True, timeout=aiohttp.ClientTimeout(total=15)
        ) as response:
            if response.status == 404:
                logger.warning("[Bing Search] - Page not found: %s", url)
                return None
            response.raise_for_status()
            return await response.text()
    except aiohttp.ClientError as e:
        logger.error("[Bing Search] - Failed to fetch page %s: %s", url, e)
        return None
    except TimeoutError:
        logger.error("[Bing Search] - Timeout while fetching page %s", url)
        return None
    except Exception as e:
        logger.error("[Bing Search] - Unexpected error fetching %s: %s", url, e)
        return None


async def fetch_search_results(
    session: aiohttp.ClientSession, headers: dict[str, str], params: dict[str, str]
) -> dict[str, Any]:
    """
    Fetches search results from the Bing Search API.

    This asynchronous function sends a GET request to the Bing Search API endpoint
    to retrieve search results based on the provided headers and query parameters.

    Args:
        session: The aiohttp session used to make the HTTP request.
        headers: A dictionary containing the request headers, including
            the API key for authentication.
        params: A dictionary containing the query parameters for the search,
            such as the search query and other optional parameters.

    Returns:
        A dictionary containing the JSON response from the Bing Search API.

    Raises:
        aiohttp.ClientResponseError: If the response status indicates a client-side error.
        aiohttp.ClientError: If there is an issue with the HTTP request or response.
    """
    api_endpoint = "https://api.bing.microsoft.com/v7.0/search"

    try:
        async with session.get(
            api_endpoint, headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            response.raise_for_status()
            return await response.json(content_type=None)
    except aiohttp.ClientResponseError as e:
        logger.error("[Bing Search] - Client response error: %s", e)
        raise
    except aiohttp.ClientError as e:
        logger.error("[Bing Search] - Client error: %s", e)
        raise
    except TimeoutError as e:
        logger.error("[Bing Search] - Timeout while fetching search results")
        error_msg = f"Timeout error: {e}"
        raise aiohttp.ClientError(error_msg) from e


def process_results(
    results: list[dict[str, Any]], pages: list[str | None]
) -> list[SearchResultItem]:
    """
    Processes search results and their corresponding pages to generate a structured output.

    Args:
        results: A list of dictionaries containing search result data. Each dictionary
            may include keys such as "name", "url", and "snippet".
        pages: A list of page content corresponding to the search results. Each element
            is either a string containing the page content or None if no content is available.

    Returns:
        A list of dictionaries where each dictionary contains the following keys:
            - "title" (str): The title of the search result (from the "name" key in the result).
            - "url" (str): The URL of the search result (from the "url" key in the result).
            - "snippet" (str): A snippet or description of the search result (from the "snippet"
              key in the result).
            - "content" (str): Parsed content of the corresponding page, limited to 2000
              characters, or an empty string if no page content is available.
    """
    output = []
    max_content_length = 2000

    for i, result in enumerate(results):
        page_content = pages[i] if i < len(pages) else None
        content = parse_content(page_content, max_content_length) if page_content else ""

        output.append({
            "title": result.get("name", ""),
            "url": result.get("url", ""),
            "snippet": result.get("snippet", ""),
            "content": content,
        })
    return output


class BingSearchTool(BaseTool):
    """
    Tool for searching the web using Bing's search engine API.

    This tool fetches search results from Bing, retrieves the actual web pages,
    and returns the parsed content along with metadata.
    """

    name = "bing-search"
    description = "Search the web using Bing."
    parameters_schema: ClassVar[dict[str, Any]] = {
        "freshness": {
            "type": "string",
            "description": "Recency of results (Day, Week, Month).",
            "enum": ["Day", "Week", "Month", ""],
            "default": "",
        },
        "query": {"type": "string", "description": "Search term."},
        "user_prompt": {"type": "string", "description": "User input."},
    }
    required_parameters: ClassVar[list[str]] = ["query", "user_prompt"]

    @classmethod
    async def _run(
        cls, freshness: str = "", query: str = "", user_prompt: str = ""
    ) -> dict[str, Any]:
        """
        Executes a Bing web search and processes the results.

        Args:
            freshness: Optional filter for result recency (e.g., "Day", "Week", "Month")
            query: The search query to execute
            user_prompt: The original user query that triggered the search

        Returns:
            A dictionary containing the search results and user prompt
        """
        if not query.strip():
            logger.warning("[Bing Search] - Empty search query provided")
            return {"error": "The search query cannot be empty.", "user_prompt": user_prompt}

        if not config.bing_api_key:
            logger.error("[Bing Search] - Bing API key not configured")
            return {"error": "Bing API key not configured.", "user_prompt": user_prompt}

        timeout = aiohttp.ClientTimeout(total=60)
        headers = {
            "Ocp-Apim-Subscription-Key": config.bing_api_key.get_secret_value(),
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36 Edg/129.0.0.0"
            ),
            "Accept": "application/json",
        }

        params = {
            "q": query.strip(),
            "textDecorations": "false",
            "textFormat": "HTML",
            "count": 10,
            "responseFilter": "Webpages",
            "safeSearch": "Moderate",
        }

        if freshness and freshness in {"Day", "Week", "Month"}:
            params["freshness"] = freshness

        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                data = await fetch_search_results(session, headers, params)
            except (aiohttp.ClientError, TimeoutError) as e:
                logger.error("[Bing Search] - Failed to fetch search results: %s", e)
                return {
                    "error": "Failed to fetch search results.",
                    "user_prompt": user_prompt,
                }

            web_pages = data.get("webPages", {})
            if not isinstance(web_pages, dict):
                logger.warning("[Bing Search] - Invalid response format: webPages is not a dict")
                return {
                    "error": "Invalid response format from Bing API.",
                    "user_prompt": user_prompt,
                }

            results = web_pages.get("value", [])
            if not results:
                logger.info("[Bing Search] - No results found for query: %s", query)
                return {
                    "results": [],
                    "message": "No results found for the query.",
                    "user_prompt": user_prompt,
                }

            # Use asyncio.gather to fetch all pages in parallel with timeout handling
            fetch_tasks = [
                fetch_page(session, r.get("url", ""))
                for r in results
                if r.get("url") and isinstance(r.get("url"), str)
            ]

            if not fetch_tasks:
                return {
                    "error": "No valid URLs found in the results.",
                    "user_prompt": user_prompt,
                }

            pages = await asyncio.gather(*fetch_tasks, return_exceptions=False)

            # Check if we have any successful page fetches
            valid_pages = [p for p in pages if p is not None]
            if not valid_pages:
                logger.warning(
                    "[Bing Search] - Could not fetch any pages from the search results."
                )
                return {
                    "error": "Could not fetch content from the result pages.",
                    "user_prompt": user_prompt,
                }

            output = process_results(results, pages)
            return {"results": output, "user_prompt": user_prompt}


tool_manager.register_tool(BingSearchTool)
