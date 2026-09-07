from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AskRequest(StrictModel):
    question: str = Field(min_length=1)
    limit: int | None = Field(default=None, ge=1, le=100)


class RetrievedDocument(StrictModel):
    identifier: str
    text: str
    metadata: dict[str, str | int]
    distance: float | None = None


class AskResponse(StrictModel):
    answer: str
    sources: list[RetrievedDocument]


class QueryResponse(StrictModel):
    ids: list[list[str]]
    distances: list[list[float]]


class SearchToolParams(StrictModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=100)


T = TypeVar("T")


class ToolResult(BaseModel, Generic[T]):
    model_config = ConfigDict(extra="forbid")

    success: bool
    data: T | None = None
    error: str | None = None
