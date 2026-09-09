from pathlib import Path
from unittest.mock import Mock

import pytest

from researchpal.config import Settings
from researchpal.models import ChunkMetadata
from researchpal.pipeline import ingestion


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
    monkeypatch.setattr(ingestion, "VectorStore", lambda *args: store)

    result = ingestion.ingest_papers(settings=settings)

    assert result == list(ingestion.REQUIRED_ARXIV_IDS)
    store.upsert.assert_called_once()
    assert len(store.upsert.call_args.args[0]) > 0
