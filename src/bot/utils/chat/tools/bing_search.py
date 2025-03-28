# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

import aiohttp
import orjson

from bot import config


async def bing_search(freshness: str = "", query: str = "") -> str:
    headers = {
        "Ocp-Apim-Subscription-Key": config.bing_api_key.get_secret_value(),
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36 Edg/129.0.0.0"
        ),
    }
    params = {"q": query, "textDecorations": "false"}
    if freshness:
        params["freshness"] = freshness

    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.get(
                "https://api.bing.microsoft.com/v7.0/search", headers=headers, params=params
            ) as response:
                response.raise_for_status()
                search_results = await response.json()
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                return "No search results found."
            raise e
        except aiohttp.ClientError as e:
            return orjson.dumps({"error": str(e)}).decode()

    output = [
        {
            "title": result.get("name", ""),
            "url": result.get("url", ""),
            "snippet": result.get("snippet", ""),
        }
        for result in search_results.get("webPages", {}).get("value", [])
    ]
    return orjson.dumps(output).decode()
