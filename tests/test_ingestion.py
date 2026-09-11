from pathlib import Path
from unittest.mock import Mock

import pytest
import tiktoken

from researchpal.config import Settings
from researchpal.models import ChunkMetadata
from researchpal.pipeline import ingestion


def test_chunk_text_splits_on_token_windows() -> None:
    encoding = tiktoken.get_encoding(ingestion.TOKEN_ENCODING)
    text = "Attention is all you need for sequence modeling and translation tasks."
    tokens = encoding.encode(text)
    chunk_size = 8
    overlap = 2
    step = chunk_size - overlap
    expected = [
        encoding.decode(tokens[start : start + chunk_size])
        for start in range(0, len(tokens), step)
    ]

    chunks = ingestion.chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    assert chunks == expected
    assert chunks != [
        text.strip()[start : start + chunk_size]
        for start in range(0, len(text.strip()), step)
    ]


def test_chunk_text_returns_empty_list_for_blank_input() -> None:
    assert ingestion.chunk_text("   ") == []


def test_chunk_text_rejects_invalid_window_sizes() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        ingestion.chunk_text("attention", chunk_size=0, overlap=0)
    with pytest.raises(ValueError, match="smaller than chunk_size"):
        ingestion.chunk_text("attention", chunk_size=8, overlap=8)



def test_ingest_papers_downloads_extracts_and_upserts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = Settings(
        pdf_directory=tmp_path / "pdfs",
        chroma_path=tmp_path / "chroma",
        chunk_size=20,
        chunk_overlap=5,
    )
    pdf_paths = [settings.pdf_directory / f"{paper_id}.pdf" for paper_id in ingestion.REQUIRED_ARXIV_IDS]
    monkeypatch.setattr(ingestion, "download_papers", Mock(return_value=pdf_paths))
    monkeypatch.setattr(
        ingestion,
        "read_pdf_pages",
        lambda path, paper_id: [
            ("A paper with useful evidence.", ChunkMetadata(paper_id=paper_id, page=1))
        ],
    )
    store = Mock()
    captured: dict[str, object] = {}

    def fake_vector_store(*args: object, **kwargs: object) -> Mock:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return store

    monkeypatch.setattr(ingestion, "VectorStore", fake_vector_store)

    result = ingestion.ingest_papers(settings=settings)

    assert result == list(ingestion.REQUIRED_ARXIV_IDS)
    store.upsert.assert_called_once()
    assert len(store.upsert.call_args.args[0]) > 0
    assert captured["kwargs"]["rebuild_incompatible_collection"] is True
