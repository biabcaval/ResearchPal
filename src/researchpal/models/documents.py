from pydantic import BaseModel, ConfigDict, Field


class ChunkMetadata(BaseModel):
    """Page and chunk identifiers stored with each indexed document."""

    model_config = ConfigDict(extra="ignore")

    paper_id: str
    page: int = Field(ge=1)
    chunk_index: int | None = Field(default=None, ge=0)
    chunk_size: int | None = Field(default=None, gt=0)
    chunk_overlap: int | None = Field(default=None, ge=0)

    def to_chroma(self) -> dict[str, str | int]:
        """Return only the fields Chroma accepts (no null values)."""
        return {key: value for key, value in self.model_dump().items() if value is not None}
