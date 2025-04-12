# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

import inspect
import logging
from collections.abc import Callable
from typing import Any

from azure.ai.inference.models import ChatCompletionsToolDefinition

from .base_tool import BaseTool

logger = logging.getLogger(__name__)


class ToolManager:
    """
    Manages tool registration, retrieval, and execution.

    This class provides a centralized registry of all available tools,
    facilitating discovery, validation, and execution of tools.

    Attributes:
        _tools (dict[str, Type[BaseTool]]): Registry mapping tool names to their classes.
        _instances (dict[str, BaseTool]): Cache of instantiated tool objects.

    Methods:
        register_tool: Registers a tool class.
        get_tool_definitions: Gets all tool definitions for Azure AI.
        get_tool_handlers: Gets a mapping of tool names to handler functions.
        execute_tool: Executes a tool by name with the provided arguments.
    """

    def __init__(self):
        self._tools: dict[str, type[BaseTool]] = {}
        self._instances: dict[str, BaseTool] = {}

    def register_tool(self, tool_cls: type[BaseTool]) -> None:
        """
        Registers a tool class with the manager.

        Args:
            tool_cls (Type[BaseTool]): The tool class to register.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if not inspect.isclass(tool_cls) or not issubclass(tool_cls, BaseTool):
            msg = f"Tool must be a subclass of BaseTool: {tool_cls}"
            raise TypeError(msg)

        if tool_cls.name in self._tools:
            msg = f"Tool with name '{tool_cls.name}' is already registered"
            raise ValueError(msg)

        self._tools[tool_cls.name] = tool_cls
        logger.info("Registered tool: %s", tool_cls.name)

    def get_tool_definitions(self) -> list[ChatCompletionsToolDefinition]:
        """
        Gets all tool definitions for Azure AI.

        Returns:
            list[ChatCompletionsToolDefinition]: List of tool definitions.
        """
        return [tool_cls.get_definition() for tool_cls in self._tools.values()]

    def get_tool_handlers(self) -> dict[str, Callable]:
        """
        Gets a mapping of tool names to handler functions.

        Returns:
            dict[str, Callable]: A dictionary mapping tool names to handler functions.
        """
        handlers = {}
        for name in self._tools:

            async def handler(*, tool_name: str = name, **kwargs: Any) -> Any:
                return await self.execute_tool(tool_name, **kwargs)

            handlers[name] = handler

        return handlers

    async def execute_tool(self, name: str, **kwargs: Any) -> Any:
        """
        Executes a tool by name with the provided arguments.

        Args:
            name (str): The name of the tool to execute.
            **kwargs: Arguments to pass to the tool.

        Returns:
            Any: The result of executing the tool.

        Raises:
            ValueError: If the tool name is not recognized.
        """
        if name not in self._tools:
            msg = f"Unknown tool: {name}"
            logger.error(msg)
            return {"error": msg}

        # Lazily instantiate the tool
        if name not in self._instances:
            self._instances[name] = self._tools[name]()

        # Execute the tool
        return await self._instances[name].run(**kwargs)


tool_manager = ToolManager()
