# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

import logging

import aiohttp
import orjson

logger = logging.getLogger(__name__)


async def get_github_data(
    endpoint: str, endpoint_description: str = "", repo: str = "", task: str = ""
) -> str:
    """
    Fetch data from the GitHub API for a specified endpoint.

    This asynchronous function retrieves data from the GitHub API based on the provided
    endpoint, repository, and task information. It logs the request details and handles
    errors during the API call.

    Args:
        endpoint (str): The API endpoint to fetch data from. Can include placeholders like
            "{repo}".
        endpoint_description (str, optional): A description of the endpoint being accessed.
            Defaults to an empty string.
        repo (str, optional): The repository name to replace in the endpoint if applicable.
            Defaults to an empty string.
        task (str, optional): A description of the task being performed. Defaults to an empty
            string.

    Returns:
        str: A JSON string containing the endpoint, description, repository, task, and the
            fetched data.

    Raises:
        aiohttp.ClientError: If an error occurs during the API request.
    """
    if "{repo}" in endpoint and repo:  # noqa: RUF027
        endpoint = endpoint.replace("{repo}", repo)

    url = "https://api.github.com" + endpoint
    params = {"per_page": 10, "page": 1}

    logger.info(
        "[Get GitHub Data] - Endpoint: %s | Description: %s | Repo: %s | Task: %s",
        endpoint,
        endpoint_description or "No description provided",
        repo or "No repository specified",
        task or "No task specified",
    )

    timeout = aiohttp.ClientTimeout(total=60)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            response = await session.get(url, params=params)
            response.raise_for_status()
            data = await response.json()
            result = {
                "endpoint": endpoint,
                "endpoint_description": endpoint_description,
                "repo": repo,
                "task": task,
                "data": data,
            }
            return orjson.dumps(result).decode()
    except aiohttp.ClientError as e:
        logger.exception("[Get GitHub Data] - Failed to fetch data: %s", e)
        raise
