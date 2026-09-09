import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import chromadb
from chromadb.api.types import EmbeddingFunction
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from pydantic import ValidationError

from researchpal.models import (
    ChromaQueryResult,
    ChunkMetadata,
    QueryResponse,
    RetrievedDocument,
)

logger = logging.getLogger(__name__)


class VectorCollection(Protocol):
    """Chroma-compatible collection used by VectorStore."""

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, str | int]],
    ) -> None: ...

    def query(self, query_texts: list[str], n_results: int) -> object: ...


class VectorStore:
    def __init__(
        self,
        path: Path,
        collection_name: str = "papers",
        embedding_function: EmbeddingFunction | None = None,
        collection: VectorCollection | None = None,
    ) -> None:
        if collection is not None:
            self.collection = collection
            return

        self.client = chromadb.PersistentClient(path=str(path))
        self.embedding_function = embedding_function or DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
        )

    def upsert(
        self,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[ChunkMetadata],
    ) -> None:
        self.collection.upsert(
            ids=list(ids),
            documents=list(documents),
            metadatas=[item.to_chroma() for item in metadatas],
        )

    def query(self, question: str, limit: int = 5) -> QueryResponse:
        parsed = parse_chroma_query(
            self.collection.query(query_texts=[question], n_results=limit)
        )
        return QueryResponse(ids=parsed.ids, distances=parsed.distances)

    def search(self, question: str, limit: int = 5) -> list[RetrievedDocument]:
        parsed = parse_chroma_query(
            self.collection.query(query_texts=[question], n_results=limit)
        )
        ids = parsed.batch_ids()
        documents = parsed.batch_documents()
        metadatas = parsed.batch_metadatas()
        distances = parsed.batch_distances()

        return [
            RetrievedDocument(
                identifier=identifier,
                text=document or "",
                metadata=_chunk_metadata(identifier, metadata),
                distance=distances[index] if index < len(distances) else None,
            )
            for index, (identifier, document, metadata) in enumerate(
                zip(ids, documents, metadatas, strict=False)
            )
        ]


def parse_chroma_query(payload: object) -> ChromaQueryResult:
    """Validate a Chroma query dict into a typed result.

    Args:
        payload: Raw mapping returned by `collection.query`.

    Returns:
        Nested ids, documents, metadatas, and distances with missing fields empty.
    """
    return ChromaQueryResult.model_validate(payload or {})


def _chunk_metadata(identifier: str, metadata: object) -> ChunkMetadata:
    try:
        return ChunkMetadata.model_validate(metadata or {"paper_id": identifier, "page": 1})
    except ValidationError:
        logger.warning(
            "Invalid chunk metadata for %s; using fallback paper_id/page",
            identifier,
        )
        return ChunkMetadata(paper_id=identifier, page=1)
