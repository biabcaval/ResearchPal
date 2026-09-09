"""Local PDF page extraction shared by tools and ingestion."""

from pathlib import Path

from PyPDF2 import PdfReader

from researchpal.models import ChunkMetadata


def read_pdf_pages(filepath: Path, paper_id: str) -> list[tuple[str, ChunkMetadata]]:
    """Return `(text, metadata)` for each page of a local PDF."""
    reader = PdfReader(filepath)
    pages: list[tuple[str, ChunkMetadata]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        pages.append(
            (
                page.extract_text() or "",
                ChunkMetadata(paper_id=paper_id, page=page_number),
            )
        )
    return pages
