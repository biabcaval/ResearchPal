from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from researchpal.models.documents import ChunkMetadata


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AskRequest(StrictModel):
    question: str = Field(min_length=1)


class RetrievedDocument(StrictModel):
    identifier: str
    text: str
    metadata: ChunkMetadata
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


class ChromaQueryResult(BaseModel):
    """Typed Chroma `collection.query` payload after the SDK dict boundary."""

    model_config = ConfigDict(extra="ignore")

    ids: list[list[str]] = Field(default_factory=list)
    documents: list[list[str | None]] = Field(default_factory=list)
    metadatas: list[list[object | None]] = Field(default_factory=list)
    distances: list[list[float]] = Field(default_factory=list)

    @field_validator("ids", "documents", "metadatas", "distances", mode="before")
    @classmethod
    def empty_when_missing(cls, value: object) -> object:
        return [] if value is None else value

    def batch_ids(self) -> list[str]:
        """Return the first query batch of document ids."""
        return self.ids[0] if self.ids else []

    def batch_documents(self) -> list[str | None]:
        """Return the first query batch of document texts."""
        return self.documents[0] if self.documents else []

    def batch_metadatas(self) -> list[object | None]:
        """Return the first query batch of raw Chroma metadata rows."""
        return self.metadatas[0] if self.metadatas else []

    def batch_distances(self) -> list[float]:
        """Return the first query batch of distances."""
        return self.distances[0] if self.distances else []


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
