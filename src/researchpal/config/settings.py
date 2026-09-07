from pathlib import Path

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REQUIRED_ARXIV_IDS = ("1706.03762", "1810.04805", "2005.11401")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("GEMINI_API_KEY", "gemini_api_key"),
    )
    gemini_model: str = Field(
        default="gemini-3.5-flash",
        validation_alias=AliasChoices("GEMINI_MODEL", "gemini_model"),
    )

    chroma_path: Path = Field(
        default=PROJECT_ROOT / "data" / "chroma",
        validation_alias=AliasChoices("RESEARCHPAL_CHROMA_PATH", "chroma_path"),
    )
    collection_name: str = Field(
        default="papers",
        validation_alias=AliasChoices("RESEARCHPAL_COLLECTION", "collection_name"),
    )
    pdf_directory: Path = Field(
        default=PROJECT_ROOT / "data" / "pdfs",
        validation_alias=AliasChoices("RESEARCHPAL_PDF_DIRECTORY", "pdf_directory"),
    )
    request_timeout: int = Field(
        default=30,
        ge=1,
        validation_alias=AliasChoices("RESEARCHPAL_REQUEST_TIMEOUT", "request_timeout"),
    )
    download_chunk_size: int = Field(
        default=1024 * 1024,
        gt=0,
        validation_alias=AliasChoices(
            "RESEARCHPAL_DOWNLOAD_CHUNK_SIZE",
            "download_chunk_size",
        ),
    )
    chunk_size: int = Field(
        default=1000,
        gt=0,
        validation_alias=AliasChoices("RESEARCHPAL_CHUNK_SIZE", "chunk_size"),
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        validation_alias=AliasChoices("RESEARCHPAL_CHUNK_OVERLAP", "chunk_overlap"),
    )
    retrieval_limit: int = Field(
        default=5,
        ge=1,
        le=100,
        validation_alias=AliasChoices("RESEARCHPAL_RETRIEVAL_LIMIT", "retrieval_limit"),
    )
    retrieval_score_threshold: float | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices(
            "RESEARCHPAL_RETRIEVAL_SCORE_THRESHOLD",
            "retrieval_score_threshold",
        ),
    )

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


def get_settings() -> Settings:
    return Settings()
