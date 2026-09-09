from pathlib import Path
from typing import Any

import pytest

from researchpal.models import ChunkMetadata
from researchpal.tools.vector_store import VectorStore, parse_chroma_query


class FakeCollection:
    def __init__(self) -> None:
        self.upserts: list[tuple[list[str], list[str], list[dict[str, str | int]]]] = []
        self.query_result: dict[str, Any] = {
            "ids": [["1706.03762-page-1-chunk-0"]],
            "documents": [["attention is all you need"]],
            "metadatas": [
                [{"paper_id": "1706.03762", "page": 1, "chunk_index": 0, "extra": "ignored"}]
            ],
            "distances": [[0.12]],
        }

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, str | int]],
    ) -> None:
        self.upserts.append((ids, documents, metadatas))

    def query(self, query_texts: list[str], n_results: int) -> dict[str, Any]:
        return self.query_result


def test_parse_chroma_query_maps_nested_lists_and_ignores_sdk_fields() -> None:
    parsed = parse_chroma_query(
        {
            "ids": [["chunk-0"]],
            "documents": [["attention"]],
            "metadatas": [[{"paper_id": "1706.03762", "page": 1}]],
            "distances": [[0.12]],
            "embeddings": [[[0.1, 0.2]]],
        }
    )

    assert parsed.batch_ids() == ["chunk-0"]
    assert parsed.batch_documents() == ["attention"]
    assert parsed.batch_metadatas() == [{"paper_id": "1706.03762", "page": 1}]
    assert parsed.batch_distances() == [0.12]


def test_parse_chroma_query_treats_missing_or_null_fields_as_empty_batches() -> None:
    parsed = parse_chroma_query({"ids": None, "documents": None})

    assert parsed.batch_ids() == []
    assert parsed.batch_documents() == []
    assert parsed.batch_metadatas() == []
    assert parsed.batch_distances() == []


def test_upsert_sends_chroma_dicts_without_null_fields() -> None:
    collection = FakeCollection()
    store = VectorStore(path=Path("unused"), collection=collection)
    metadata = ChunkMetadata(
        paper_id="1706.03762",
        page=1,
        chunk_index=0,
        chunk_size=1000,
        chunk_overlap=200,
    )

    store.upsert(["id-1"], ["chunk text"], [metadata])

    assert collection.upserts == [
        (["id-1"], ["chunk text"], [metadata.to_chroma()])
    ]
    assert None not in collection.upserts[0][2][0].values()


def test_search_maps_collection_rows_to_retrieved_documents() -> None:
    collection = FakeCollection()
    store = VectorStore(path=Path("unused"), collection=collection)

    results = store.search("attention", limit=5)

    assert len(results) == 1
    assert results[0].identifier == "1706.03762-page-1-chunk-0"
    assert results[0].text == "attention is all you need"
    assert results[0].metadata.paper_id == "1706.03762"
    assert results[0].metadata.page == 1
    assert results[0].metadata.chunk_index == 0
    assert results[0].distance == 0.12


def test_query_returns_ids_and_distances_from_the_collection() -> None:
    collection = FakeCollection()
    store = VectorStore(path=Path("unused"), collection=collection)

    result = store.query("attention", limit=3)

    assert result.ids == [["1706.03762-page-1-chunk-0"]]
    assert result.distances == [[0.12]]


def test_search_falls_back_when_metadata_is_invalid(
    caplog: pytest.LogCaptureFixture,
) -> None:
    collection = FakeCollection()
    collection.query_result = {
        "ids": [["orphan"]],
        "documents": [["text"]],
        "metadatas": [[{"page": 0}]],
        "distances": [[0.5]],
    }
    store = VectorStore(path=Path("unused"), collection=collection)

    with caplog.at_level("WARNING", logger="researchpal.tools.vector_store"):
        results = store.search("query")

    assert results[0].metadata.paper_id == "orphan"
    assert results[0].metadata.page == 1
    assert "Invalid chunk metadata for orphan" in caplog.text
