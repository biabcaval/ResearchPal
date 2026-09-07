from pathlib import Path
from unittest.mock import Mock

from researchpal.models import ExtractSectionParams, RetrievedDocument, SearchToolParams
from researchpal.tools.document_tools import extract_section, search_documents
from researchpal.tools import document_tools
from researchpal.config import Settings


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
        "researchpal.pipeline.ingestion.read_pdf_pages",
        lambda path, paper_id: [
            ("Abstract\nUseful result\nIntroduction\nMore context", {"page": 1})
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
