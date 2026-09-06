from collections.abc import Sequence
from pathlib import Path
from typing import Any

import chromadb

# Abs for creating/accessing persistent chromadb client

class VectorStore:
    def __init__(self, path: Path, collection_name: str = "papers") -> None:
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection = self.client.get_or_create_collection(name=collection_name)

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

    def query(self, question: str, limit: int = 5) -> dict[str, Any]:
        return self.collection.query(query_texts=[question], n_results=limit)
