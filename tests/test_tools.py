from pathlib import Path
from unittest.mock import Mock

import pytest

from researchpal.config import Settings
from researchpal.models import (
    ChunkMetadata,
    ExtractSectionParams,
    RetrievedDocument,
    SearchToolParams,
)
from researchpal.tools import document_tools
from researchpal.tools.document_tools import extract_section, search_documents


def test_search_documents_uses_the_vector_store() -> None:
    expected = [
        RetrievedDocument(
            identifier="1706.03762-page-1-chunk-0",
            text="attention is all you need",
            metadata={"paper_id": "1706.03762", "page": 1},
            distance=0.1,
        )
    ]
    store = Mock()
    store.search.return_value = expected

    result = search_documents(SearchToolParams(query="attention"), store=store)

    assert result.success is True
    assert result.data == expected
    store.search.assert_called_once_with("attention", 5)


def test_search_documents_rewrites_the_query_to_english_before_chroma(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = Mock()
    store.search.return_value = []
    monkeypatch.setattr(
        document_tools,
        "to_english_retrieval_query",
        lambda query, settings=None: "central mechanism self-attention",
    )

    result = search_documents(
        SearchToolParams(query="Qual é o mecanismo central?"),
        store=store,
    )

    assert result.success is True
    store.search.assert_called_once_with("central mechanism self-attention", 5)


def test_extract_section_reads_the_requested_pdf(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "1706.03762.pdf"
    pdf_path.write_bytes(b"pdf")
    settings = Settings(pdf_directory=tmp_path)
    monkeypatch.setattr(
        document_tools,
        "get_settings",
        lambda: settings,
    )
    monkeypatch.setattr(
        document_tools,
        "read_pdf_pages",
        lambda path, paper_id: [
            (
                "Abstract\nUseful result\nIntroduction\nMore context",
                ChunkMetadata(paper_id=paper_id, page=1),
            )
        ],
    )

    result = extract_section(
        ExtractSectionParams(paper_id="1706.03762", section="abstract"),
        settings=settings,
    )

    assert result.success is True
    assert result.data is not None
    assert result.data.text == "Useful result"


def test_extract_section_rejects_unknown_paper(tmp_path: Path) -> None:
    result = extract_section(
        ExtractSectionParams(paper_id="unknown", section="abstract"),
        settings=Settings(pdf_directory=tmp_path),
    )

    assert result.success is False
    assert result.error == "Unsupported paper_id: unknown"


def test_search_documents_logs_vector_store_failures(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = Mock()
    store.search.side_effect = RuntimeError("chroma is down")

    with caplog.at_level("WARNING", logger="researchpal.tools.document_tools"):
        result = search_documents(SearchToolParams(query="attention"), store=store)

    assert result.success is False
    assert "chroma is down" in (result.error or "")
    assert "Document search failed" in caplog.text
