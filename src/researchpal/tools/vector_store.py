from collections.abc import Sequence
from pathlib import Path

import chromadb
from chromadb.api.types import EmbeddingFunction
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from researchpal.models import QueryResponse, RetrievedDocument


class VectorStore:
    def __init__(
        self,
        path: Path,
        collection_name: str = "papers",
        embedding_function: EmbeddingFunction | None = None,
    ) -> None:
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
        metadatas: Sequence[dict[str, str | int]],
    ) -> None:
        self.collection.upsert(
            ids=list(ids),
            documents=list(documents),
            metadatas=list(metadatas),
        )

    def query(self, question: str, limit: int = 5) -> QueryResponse:
        result = self.collection.query(query_texts=[question], n_results=limit)
        return QueryResponse(
            ids=result.get("ids", []),
            distances=result.get("distances", []),
        )

    def search(self, question: str, limit: int = 5) -> list[RetrievedDocument]:
        result = self.collection.query(query_texts=[question], n_results=limit)
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        return [
            RetrievedDocument(
                identifier=identifier,
                text=document or "",
                metadata=metadata or {},
                distance=distances[index] if index < len(distances) else None,
            )
            for index, (identifier, document, metadata) in enumerate(
                zip(ids, documents, metadatas, strict=False)
            )
        ]
