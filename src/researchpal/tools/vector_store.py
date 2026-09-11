import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import chromadb
from chromadb.api.types import EmbeddingFunction
from pydantic import ValidationError

from researchpal.models import (
    ChromaQueryResult,
    ChunkMetadata,
    QueryResponse,
    RetrievedDocument,
)
from researchpal.tools.embeddings import paper_embedding_function

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
        rebuild_incompatible_collection: bool = False,
    ) -> None:
        if collection is not None:
            self.collection = collection
            return

        self.client = chromadb.PersistentClient(path=str(path))
        self.embedding_function = embedding_function or paper_embedding_function()
        self.collection = self._get_or_create_collection(
            collection_name,
            rebuild_incompatible_collection=rebuild_incompatible_collection,
        )

    def _get_or_create_collection(
        self,
        collection_name: str,
        *,
        rebuild_incompatible_collection: bool,
    ) -> VectorCollection:
        """Open the collection, optionally replacing one built with another encoder."""
        try:
            collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
            )
        except ValueError as error:
            if not _is_embedding_function_conflict(error):
                raise
            return self._rebuild_or_raise(
                collection_name,
                rebuild_incompatible_collection,
                error,
            )

        if _encoder_names_differ(collection, self.embedding_function):
            return self._rebuild_or_raise(
                collection_name,
                rebuild_incompatible_collection,
                None,
            )
        return collection

    def _rebuild_or_raise(
        self,
        collection_name: str,
        rebuild_incompatible_collection: bool,
        error: ValueError | None,
    ) -> VectorCollection:
        """Delete and recreate the collection, or tell the caller to re-ingest."""
        if not rebuild_incompatible_collection:
            raise RuntimeError(
                "The Chroma collection was indexed with a different embedding "
                "function. Re-run `uv run python ingest.py` to rebuild the index."
            ) from error
        logger.warning(
            "Replacing collection %s after embedding-function change",
            collection_name,
        )
        self.client.delete_collection(collection_name)
        return self.client.get_or_create_collection(
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


def _is_embedding_function_conflict(error: ValueError) -> bool:
    """Return True when Chroma rejected a collection because the encoder changed."""
    return "embedding function conflict" in str(error).lower()


def _encoder_names_differ(collection: object, embedding_function: object) -> bool:
    """Return True when the open collection was built with another encoder name.

    Chroma skips the conflict check when the new function is named ``default``,
    so switching back from sentence-transformers MiniLM would otherwise mix
    384-d vectors from two different spaces.
    """
    persisted = _persisted_embedding_name(collection)
    current = _embedding_function_name(embedding_function)
    if persisted is None or current is None:
        return False
    return persisted != current


def _persisted_embedding_name(collection: object) -> str | None:
    """Read the encoder name stored on a Chroma collection, if present."""
    config = getattr(collection, "configuration_json", None)
    if config is None:
        configuration = getattr(collection, "configuration", None)
        if isinstance(configuration, dict):
            config = configuration
        else:
            model = getattr(collection, "_model", None)
            config = getattr(model, "configuration_json", None) if model is not None else None
    if not isinstance(config, dict):
        return None
    embedding = config.get("embedding_function")
    if not isinstance(embedding, dict):
        return None
    name = embedding.get("name")
    return name if isinstance(name, str) and name else None


def _embedding_function_name(embedding_function: object) -> str | None:
    """Return the Chroma encoder name, or None when the object has no name()."""
    name = getattr(embedding_function, "name", None)
    if not callable(name):
        return None
    resolved = name()
    return resolved if isinstance(resolved, str) and resolved else None
