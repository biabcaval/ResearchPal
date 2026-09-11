import logging
import re

import chromadb
from PyPDF2.errors import PdfReadError

from researchpal.config import REQUIRED_ARXIV_IDS, Settings, get_settings
from researchpal.models import (
    ExtractedSection,
    ExtractSectionParams,
    RetrievedDocument,
    SearchToolParams,
    ToolResult,
)
from researchpal.tools.english_retrieval import to_english_retrieval_query
from researchpal.tools.pdf import read_pdf_pages
from researchpal.tools.vector_store import VectorStore

logger = logging.getLogger(__name__)

ALLOWED_SECTIONS = frozenset({"abstract", "introduction", "conclusion"})
SEARCH_DOCUMENTS_DESCRIPTION = (
    "Search indexed paper chunks semantically by query and return the most relevant "
    "documents up to the requested limit. The index is English, so the query is "
    "rewritten to English before comparison. Answers to the user stay in Portuguese."
)
EXTRACT_SECTION_DESCRIPTION = (
    "Extract the abstract, introduction, or conclusion from one supported paper PDF."
)
SECTION_HEADINGS = {
    "abstract": re.compile(r"^\s*(?:abstract)\s*:?\s*$", re.IGNORECASE),
    "introduction": re.compile(
        r"^\s*(?:\d+(?:\.\d+)*\s+)?introduction\s*$", re.IGNORECASE
    ),
    "conclusion": re.compile(
        r"^\s*(?:\d+(?:\.\d+)*\s+)?conclusions?\s*$", re.IGNORECASE
    ),
}


def search_documents(
    params: SearchToolParams,
    settings: Settings | None = None,
    store: VectorStore | None = None,
) -> ToolResult[list[RetrievedDocument]]:
    """Search indexed chunks after rewriting the query to English for MiniLM."""
    active_settings = settings or get_settings()
    active_store = store or VectorStore(
        active_settings.chroma_path,
        active_settings.collection_name,
    )
    try:
        retrieval_query = to_english_retrieval_query(
            params.query,
            settings=active_settings,
        )
        documents = active_store.search(retrieval_query, params.limit)
    except (chromadb.errors.ChromaError, RuntimeError, ValueError) as error:
        logger.warning("Document search failed: %s", error)
        return ToolResult(success=False, error=f"Document search failed: {error}")
    return ToolResult(success=True, data=documents)


def extract_section(
    params: ExtractSectionParams,
    settings: Settings | None = None,
) -> ToolResult[ExtractedSection]:
    """Extract one supported paper section from its locally stored PDF."""
    active_settings = settings or get_settings()
    paper_id = params.paper_id.strip()
    section = params.section

    if paper_id not in REQUIRED_ARXIV_IDS:
        return ToolResult(
            success=False,
            error=f"Unsupported paper_id: {paper_id}",
        )
    if section not in ALLOWED_SECTIONS:
        return ToolResult(
            success=False,
            error=f"Unsupported section: {section}",
        )

    pdf_path = active_settings.pdf_directory / f"{paper_id}.pdf"
    if not pdf_path.is_file() or pdf_path.stat().st_size == 0:
        return ToolResult(
            success=False,
            error=f"PDF is not available locally for paper_id: {paper_id}",
        )

    try:
        pages = read_pdf_pages(pdf_path, paper_id)
    except (OSError, PdfReadError, ValueError) as error:
        logger.warning("Failed to read PDF for paper_id %s: %s", paper_id, error)
        return ToolResult(success=False, error=f"Failed to read PDF: {error}")

    text = "\n".join(page_text for page_text, _ in pages).strip()
    extracted = _find_section(text, section)
    if extracted is None:
        return ToolResult(
            success=False,
            error=f"Section '{section}' was not found in paper_id: {paper_id}",
        )

    return ToolResult(
        success=True,
        data=ExtractedSection(paper_id=paper_id, section=section, text=extracted),
    )


def _find_section(text: str, section: str) -> str | None:
    lines = text.splitlines()
    start: int | None = None
    for index, line in enumerate(lines):
        if SECTION_HEADINGS[section].match(line.strip()):
            start = index + 1
            break

    if start is None:
        return None

    for index in range(start, len(lines)):
        if any(pattern.match(lines[index].strip()) for pattern in SECTION_HEADINGS.values()):
            result = "\n".join(lines[start:index]).strip()
            return result or None

    result = "\n".join(lines[start:]).strip()
    return result or None
