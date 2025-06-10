# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

import logging
from typing import Any, ClassVar

import aiohttp

from .base_tool import BaseTool
from .tool_manager import tool_manager

logger = logging.getLogger(__name__)


class GitHubDataTool(BaseTool):
    """
    Tool for fetching data from the GitHub API.

    This tool provides access to GitHub's REST API endpoints
    for retrieving repository data and other GitHub resources.
    """

    name = "get-github-data"
    description = (
        "This tool provides GET-only access to GitHub's REST API, enabling structured queries "
        "for GitHub resources like repositories, issues, pull requests, and content."
    )
    parameters_schema: ClassVar[dict[str, Any]] = {
        "endpoint": {
            "type": "string",
            "description": (
                "A full valid GitHub REST API endpoint to call via a GET request. "
                "Include the leading slash."
            ),
        },
        "endpoint_description": {
            "type": "string",
            "description": (
                "A short description of the GitHub API operation. This should be generic, and "
                "not mention any particular entities. For example, 'get repo' or 'search pull "
                "requests' or 'list releases in repo'. Prefer 'search' over 'list' for issues "
                "and pull requests."
            ),
            "default": "",
        },
        "repo": {
            "type": "string",
            "description": (
                "The 'owner/repo' name of the repository that's being used in the endpoint. "
                "If this isn't used in the endpoint, send an empty string."
            ),
            "default": "",
        },
        "task": {
            "type": "string",
            "description": (
                "A phrase describing the task to be accomplished with the GitHub REST API. "
                "For example, 'search for issues assigned to user monalisa' or 'get pull "
                "request number 42 in repo facebook/react' or 'list releases in repo "
                "kubernetes/kubernetes'. If the user is asking about data in a particular "
                "repo, that repo should be specified."
            ),
            "default": "",
        },
    }
    required_parameters: ClassVar[list[str]] = ["endpoint", "repo"]

    @classmethod
    async def _run(
        cls, endpoint: str, endpoint_description: str = "", repo: str = "", task: str = ""
    ) -> dict[str, Any]:
        """
        Executes a request to the GitHub API.

        Args:
            endpoint: The API endpoint to fetch data from. Can include placeholders like "{repo}".
            endpoint_description: A description of the endpoint being accessed.
            repo: The repository name to replace in the endpoint if applicable.
            task: A description of the task being performed.

        Returns:
            A dictionary containing the endpoint, description, repo, task, and fetched data.
        """
        if "{repo}" in endpoint and repo:  # noqa: RUF027
            endpoint = endpoint.replace("{repo}", repo)

        url = "https://api.github.com" + endpoint
        params = {"per_page": 10, "page": 1}

        logger.info(
            "[GitHub Data] - Endpoint: %s | Description: %s | Repo: %s | Task: %s",
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
                return {
                    "endpoint": endpoint,
                    "endpoint_description": endpoint_description,
                    "repo": repo,
                    "task": task,
                    "data": data,
                }
        except aiohttp.ClientError as e:
            logger.exception("[GitHub Data] - Failed to fetch data: %s", e)
            return {
                "error": f"Failed to fetch data from GitHub API: {e!s}",
                "endpoint": endpoint,
                "repo": repo,
            }


tool_manager.register_tool(GitHubDataTool)
