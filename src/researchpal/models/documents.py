from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class Document:
    identifier: str
    text: str


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=100)


@dataclass(frozen=True)
class QueryResult:
    ids: list[list[str]]
    distances: list[list[float]]
    raw: dict[str, Any]
