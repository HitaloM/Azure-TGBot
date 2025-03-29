# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

import asyncio
import logging
import re
from typing import Any

import aiohttp
import orjson
from lxml import html

from bot import config

logger = logging.getLogger(__name__)


def parse_content(html_content: str, max_characters: int | None = None) -> str:
    """
    Parses the given HTML content and extracts readable text while removing unnecessary elements.

    This function removes script, style, header, footer, navigation, and aside elements from
    the HTML content, then extracts the text content from the body. It also cleans up non-ASCII
    characters and excessive whitespace.

    Args:
        html_content (str): The HTML content to parse.
        max_characters (int | None, optional): The maximum number of characters to include in
            the output. If None, the entire cleaned text is returned. Defaults to None.

    Returns:
        str: The cleaned and extracted text content from the HTML.
    """
    tree = html.fromstring(html_content)
    for element in tree.xpath("//script | //style | //header | //footer | //nav | //aside"):
        parent = element.getparent()
        if parent is not None:
            parent.remove(element)

    text = tree.xpath("string(body)")
    text = re.sub(r"[^\x00-\x7F]+|\s+", " ", text)
    return text[:max_characters].strip() if max_characters else text.strip()


async def fetch_page(session: aiohttp.ClientSession, url: str) -> str | None:
    """
    Fetches the content of a web page asynchronously.

    Args:
        session (aiohttp.ClientSession): The aiohttp session to use for making the HTTP request.
        url (str): The URL of the web page to fetch.

    Returns:
        str | None: The content of the web page as a string if the request is successful,
        or None if the URL is invalid, the page is not found (404), or an error occurs during the
        request.
    """
    if not url:
        return None

    try:
        async with session.get(url) as response:
            if response.status == 404:
                logger.warning("Page not found: %s", url)
                return None
            response.raise_for_status()
            return await response.text()
    except aiohttp.ClientError as e:
        logger.error("[Bing Search] - Failed to fetch page %s: %s", url, e)
        return None


async def fetch_search_results(
    session: aiohttp.ClientSession, headers: dict[str, str], params: dict[str, str]
) -> dict[str, Any]:
    """
    Fetches search results from the Bing Search API.

    This asynchronous function sends a GET request to the Bing Search API endpoint
    to retrieve search results based on the provided headers and query parameters.

    Args:
        session (aiohttp.ClientSession): The aiohttp session used to make the HTTP request.
        headers (dict[str, str]): A dictionary containing the request headers, including
            the API key for authentication.
        params (dict[str, str]): A dictionary containing the query parameters for the search,
            such as the search query and other optional parameters.

    Returns:
        dict[str, Any]: A dictionary containing the JSON response from the Bing Search API.

    Raises:
        aiohttp.ClientResponseError: If the response status indicates a client-side error.
        aiohttp.ClientError: If there is an issue with the HTTP request or response.
    """
    try:
        async with session.get(
            "https://api.bing.microsoft.com/v7.0/search", headers=headers, params=params
        ) as response:
            response.raise_for_status()
            return await response.json()
    except aiohttp.ClientResponseError as e:
        logger.error("[Bing Search] - Client response error: %s", e)
        raise
    except aiohttp.ClientError as e:
        logger.error("[Bing Search] - Client error: %s", e)
        raise


def process_results(results: list, pages: list) -> list:
    """
    Processes search results and their corresponding pages to generate a structured output.

    Args:
        results (list): A list of dictionaries containing search result data. Each dictionary
            may include keys such as "name", "url", and "snippet".
        pages (list): A list of page content corresponding to the search results. Each element
            is either a string containing the page content or None if no content is available.

    Returns:
        list: A list of dictionaries where each dictionary contains the following keys:
              - "title" (str): The title of the search result (from the "name" key in the result).
              - "url" (str): The URL of the search result (from the "url" key in the result).
              - "snippet" (str): A snippet or description of the search result (from the "snippet"
                key in the result).
              - "content" (str): Parsed content of the corresponding page, limited to 2000
                characters, or an empty string if no page content is available.
    """
    output = []
    for result, page in zip(results, pages, strict=False):
        content = parse_content(page, 2000) if page else ""
        output.append({
            "title": result.get("name", ""),
            "url": result.get("url", ""),
            "snippet": result.get("snippet", ""),
            "content": content,
        })
    return output


async def bing_search(freshness: str = "", query: str = "", user_prompt: str = "") -> str:
    """
    Performs a Bing search query and processes the results.

    Args:
        freshness (str, optional): Specifies the freshness of the search results.
            Possible values include "Day", "Week", or "Month". Defaults to an empty string.
        query (str, optional): The search query string. Defaults to an empty string.
        user_prompt (str, optional): A user-provided prompt to include in the result object.
            Defaults to an empty string.

    Returns:
        str: A JSON-encoded string containing the processed search results and the user prompt.
            If an error occurs during the search, a JSON-encoded error message is returned.

    Raises:
        aiohttp.ClientError: If there is an issue with the HTTP request during the search process.

    Notes:
        - The function uses the Bing Search API to fetch search results.
        - The `freshness` parameter is optional and filters results based on their recency.
        - The function processes the search results by fetching additional page data for each
            result URL.
        - The `config.bing_api_key` is required to authenticate with the Bing Search API.
    """
    timeout = aiohttp.ClientTimeout(total=60)
    headers = {
        "Ocp-Apim-Subscription-Key": config.bing_api_key.get_secret_value(),
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36 Edg/129.0.0.0"
        ),
    }
    params = {"q": query, "textDecorations": "false", "count": 10}
    if freshness:
        params["freshness"] = freshness

    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            data = await fetch_search_results(session, headers, params)
        except aiohttp.ClientError:
            logger.error("[Bing Search] - Failed to fetch search results.")
            return orjson.dumps({"error": "Failed to fetch search results."}).decode()

        web_pages = data.get("webPages", {})
        results = web_pages.get("value", []) if isinstance(web_pages, dict) else []
        pages = await asyncio.gather(*[
            fetch_page(session, r.get("url", "")) for r in results if r.get("url")
        ])

        if all(page is None for page in pages):
            logger.error("[Bing Search] - Could not fetch any pages from the search results.")
            return orjson.dumps({
                "error": "Could not fetch any pages from the search results."
            }).decode()

        output = process_results(results, pages)
        result_obj = {"results": output, "user_prompt": user_prompt}
        return orjson.dumps(result_obj).decode()
