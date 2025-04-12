# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

import inspect
import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar, TypeVar

from azure.ai.inference.models import ChatCompletionsToolDefinition, FunctionDefinition
from pydantic import BaseModel, create_model

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="BaseTool")


class BaseTool(ABC):
    """
    Base class for all tools/functions that can be called by the AI.

    This abstract class provides the structure and interface for all tools
    that can be invoked through function calling.

    Attributes:
        name (ClassVar[str]): The name of the tool as it will be recognized by the AI.
        description (ClassVar[str]): A description of what the tool does.
        parameters_schema (ClassVar[dict]): JSON schema for the tool's parameters.
        required_parameters (ClassVar[list[str]]): List of required parameter names.

    Methods:
        get_definition(): Returns the tool definition for Azure AI.
        _run(): Abstract method that implements the actual tool functionality.
        run(): Wrapper that handles parameter validation and error handling.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    parameters_schema: ClassVar[dict]
    required_parameters: ClassVar[list[str]]

    @classmethod
    def get_definition(cls) -> ChatCompletionsToolDefinition:
        """
        Returns the tool definition for Azure AI.

        Returns:
            ChatCompletionsToolDefinition: The tool definition to be registered with Azure AI.
        """
        return ChatCompletionsToolDefinition(
            function=FunctionDefinition(
                name=cls.name,
                description=cls.description,
                parameters={
                    "type": "object",
                    "properties": cls.parameters_schema,
                    "required": cls.required_parameters,
                },
            )
        )

    @classmethod
    def create_params_model(cls) -> type[BaseModel]:
        """
        Creates a Pydantic model for parameter validation based on the parameter schema.

        Returns:
            type[BaseModel]: A dynamically created Pydantic model class.
        """
        fields = {}
        for name, schema in cls.parameters_schema.items():
            annotation = str
            if schema.get("type") == "number":
                annotation = float
            elif schema.get("type") == "integer":
                annotation = int
            elif schema.get("type") == "boolean":
                annotation = bool

            default = ... if name in cls.required_parameters else schema.get("default")
            fields[name] = (annotation, default)

        return create_model(f"{cls.__name__}Params", **fields)

    @abstractmethod
    async def _run(self, **kwargs: Any) -> Any:
        """
        Implements the actual functionality of the tool.

        This method must be implemented by all subclasses.

        Args:
            **kwargs: The parameters passed to the tool.

        Returns:
            Any: The result of executing the tool.
        """
        pass

    async def run(self, **kwargs: Any) -> Any:
        """
        Executes the tool with parameter validation and error handling.

        Args:
            **kwargs: The parameters to pass to the tool.

        Returns:
            Any: The result of executing the tool.

        Raises:
            ValueError: If the parameters are invalid.
        """
        try:
            # Create and validate parameters
            params_model = self.create_params_model()
            validated_params = params_model(**kwargs)

            # Convert to dictionary
            validated_dict = validated_params.model_dump()

            # Get parameters that _run accepts
            sig = inspect.signature(self._run)
            accepted_params = {
                p.name
                for p in sig.parameters.values()
                if p.kind not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
            }

            # Filter out parameters that _run doesn't accept
            filtered_params = {k: v for k, v in validated_dict.items() if k in accepted_params}

            logger.info("Executing tool %s with parameters: %s", self.name, filtered_params)
            return await self._run(**filtered_params)
        except Exception as e:
            logger.exception("Error executing tool %s: %s", self.name, e)
            return {"error": f"Error executing {self.name}: {e!s}"}
