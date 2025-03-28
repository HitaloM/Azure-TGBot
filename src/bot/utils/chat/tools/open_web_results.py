# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

import asyncio
import re
from typing import Any

import aiohttp
from bs4 import BeautifulSoup


def parse_content(html: str, max_characters: int | None = None) -> str:
    soup = BeautifulSoup(html, "lxml")
    for element in soup(["script", "style", "header", "footer", "nav", "aside"]):
        element.decompose()

    main_content = (
        soup.find("article")
        or soup.find("div", class_=re.compile(r"(content|main|article|body)", re.IGNORECASE))
        or soup
    )

    text = main_content.get_text()[:max_characters] if max_characters else main_content.get_text()
    cleaned_text = re.sub(r"\s+", " ", text)
    cleaned_text = re.sub(r"[^\x00-\x7F]+", " ", cleaned_text).strip()

    return "\n".join(line.strip() for line in cleaned_text.splitlines() if line.strip())


async def fetch_page_async(session: aiohttp.ClientSession, url: str) -> str | None:
    try:
        async with session.get(url) as response:
            if response.status == 404:
                return None
            response.raise_for_status()
            return await response.text()
    except aiohttp.ClientResponseError as e:
        if e.status == 404:
            return None
        raise e
    except aiohttp.ClientError:
        return None


async def open_web_results(search_results: list[dict[str, str]]) -> list[dict[str, Any]] | None:
    if not search_results:
        return None

    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            fetch_page_async(session, result["url"])
            for result in search_results
            if result.get("url")
        ]
        pages = await asyncio.gather(*tasks)

    return [
        {
            "title": result["title"],
            "url": result["url"],
            "content": parse_content(html, max_characters=2000),
        }
        for result, html in zip(search_results, pages, strict=False)
        if html
    ]


async def open_url(url: str) -> dict[str, str] | str:
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        html = await fetch_page_async(session, url)
        if html:
            return {"url": url, "content": parse_content(html)}
        return "Failed to load the page."
