from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator


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


class ExtractedSection(StrictModel):
    paper_id: str
    section: str
    text: str


class AskResponse(StrictModel):
    answer: str
    sources: list[RetrievedDocument]
    sections: list[ExtractedSection] = Field(default_factory=list)
    evidence_found: bool
    tool_errors: list[str] = Field(default_factory=list)


class QueryResponse(StrictModel):
    ids: list[list[str]]
    distances: list[list[float]]


class SearchToolParams(StrictModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=100)


class ExtractSectionParams(StrictModel):
    paper_id: str = Field(min_length=1)
    section: str

    @field_validator("section")
    @classmethod
    def validate_section(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"abstract", "introduction", "conclusion"}:
            raise ValueError(
                "section must be one of: abstract, introduction, conclusion"
            )
        return normalized


T = TypeVar("T")


class ToolResult(BaseModel, Generic[T]):
    model_config = ConfigDict(extra="forbid")

    success: bool
    data: T | None = None
    error: str | None = None
