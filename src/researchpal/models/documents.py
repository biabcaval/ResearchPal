from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid")

    identifier: str
    text: str


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=100)


class QueryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ids: list[list[str]]
    distances: list[list[float]]
    raw: dict[str, Any]
