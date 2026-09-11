"""Atomic, stateless tools with a typed schema and `ToolResult` return."""

import logging
from abc import ABC, abstractmethod
from typing import ClassVar, Generic, TypeVar

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ValidationError

from researchpal.models import ToolResult

logger = logging.getLogger(__name__)

TParams = TypeVar("TParams", bound=BaseModel)
TData = TypeVar("TData")


class ResearchTool(ABC, Generic[TParams, TData]):
    """Independent tool: one operation, no memory, no workflow decisions."""

    name: ClassVar[str]
    description: ClassVar[str]
    params_model: ClassVar[type[TParams]]

    def as_langchain_tool(self) -> StructuredTool:
        """Return a LangChain tool that executes `run` and serializes `ToolResult`."""

        def _run(**kwargs: object) -> str:
            try:
                params = self.params_model.model_validate(kwargs)
            except ValidationError as error:
                return ToolResult(
                    success=False,
                    error=f"Invalid tool arguments: {error}",
                ).model_dump_json()
            try:
                result = self.run(params)
            except (OSError, RuntimeError, ValueError) as error:
                logger.warning(
                    "Tool %s execution failed: %s",
                    self.name,
                    error,
                )
                return ToolResult(
                    success=False,
                    error=f"Tool execution failed: {error}",
                ).model_dump_json()
            return result.model_dump_json()

        return StructuredTool.from_function(
            name=self.name,
            description=self.description,
            args_schema=self.params_model,
            func=_run,
            handle_validation_error=lambda error: ToolResult(
                success=False,
                error=f"Invalid tool arguments: {error}",
            ).model_dump_json(),
        )

    @abstractmethod
    def run(self, params: TParams) -> ToolResult[TData]:
        """Execute the tool with validated parameters."""
