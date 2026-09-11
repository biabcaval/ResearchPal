"""Atomic, stateless tools with a typed schema and `ToolResult` return."""

from abc import ABC, abstractmethod
from typing import ClassVar, Generic, TypeVar

from google.genai import types
from pydantic import BaseModel

from researchpal.models import ToolResult

TParams = TypeVar("TParams", bound=BaseModel)
TData = TypeVar("TData")


class ResearchTool(ABC, Generic[TParams, TData]):
    """Independent tool: one operation, no memory, no workflow decisions."""

    name: ClassVar[str]
    description: ClassVar[str]
    params_model: ClassVar[type[TParams]]

    def declaration(self) -> types.FunctionDeclaration:
        """Return the Gemini function-calling schema for this tool."""
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters_json_schema=self.params_model.model_json_schema(),
        )

    @abstractmethod
    def run(self, params: TParams) -> ToolResult[TData]:
        """Execute the tool with validated parameters."""
