"""PDF section extraction tool for the closed arXiv corpus."""

import logging
import re

from PyPDF2.errors import PdfReadError

from researchpal.config import REQUIRED_ARXIV_IDS, Settings, get_settings
from researchpal.models import ExtractedSection, ExtractSectionParams, ToolResult
from researchpal.tools.base import ResearchTool
from researchpal.tools.pdf import read_pdf_pages

logger = logging.getLogger(__name__)

ALLOWED_SECTIONS = frozenset({"abstract", "introduction", "conclusion"})
SECTION_HEADINGS = {
    "abstract": re.compile(r"^\s*(?:abstract)\s*:?\s*$", re.IGNORECASE),
    "introduction": re.compile(
        r"^\s*(?:\d+(?:\.\d+)*\s+)?introduction\s*$", re.IGNORECASE
    ),
    "conclusion": re.compile(
        r"^\s*(?:\d+(?:\.\d+)*\s+)?conclusions?\s*$", re.IGNORECASE
    ),
}


class ExtractSectionTool(ResearchTool[ExtractSectionParams, ExtractedSection]):
    """Extract abstract, introduction, or conclusion from one local PDF."""

    name = "extract_section"
    description = (
        "Extrai uma seção específica de um artigo. "
        "Use somente para abstract, introduction ou conclusion."
    )
    params_model = ExtractSectionParams

    def __init__(self, *, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def run(self, params: ExtractSectionParams) -> ToolResult[ExtractedSection]:
        """Read the local PDF and return one supported section."""
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

        pdf_path = self.settings.pdf_directory / f"{paper_id}.pdf"
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
    """Return the body of a heading-delimited section, if present."""
    lines = text.splitlines()
    start: int | None = None
    for index, line in enumerate(lines):
        if SECTION_HEADINGS[section].match(line.strip()):
            start = index + 1
            break

    if start is None:
        return None

    for index in range(start, len(lines)):
        if any(
            pattern.match(lines[index].strip()) for pattern in SECTION_HEADINGS.values()
        ):
            result = "\n".join(lines[start:index]).strip()
            return result or None

    result = "\n".join(lines[start:]).strip()
    return result or None
