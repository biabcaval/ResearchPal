from collections.abc import Sequence
from pathlib import Path

import requests
from PyPDF2 import PdfReader

from researchpal.config import REQUIRED_ARXIV_IDS, Settings, get_settings
from researchpal.tools import VectorStore


def arxiv_pdf_url(paper_id: str) -> str:
    return f"https://arxiv.org/pdf/{paper_id}.pdf"


def download_papers(
    paper_ids: Sequence[str] = REQUIRED_ARXIV_IDS,
    settings: Settings | None = None,
) -> list[Path]:
    active_settings = settings or get_settings()
    active_settings.pdf_directory.mkdir(parents=True, exist_ok=True)
    paper_paths: list[Path] = []

    for paper_id in paper_ids:
        local_path = active_settings.pdf_directory / f"{paper_id}.pdf"
        if not local_path.is_file():
            temporary_path = local_path.with_suffix(".pdf.part")
            with requests.get(
                arxiv_pdf_url(paper_id),
                stream=True,
                timeout=active_settings.request_timeout,
            ) as response:
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").split(";")[0]
                if content_type != "application/pdf":
                    raise ValueError(f"Expected a PDF from {arxiv_pdf_url(paper_id)}")
                with temporary_path.open("wb") as pdf_file:
                    for chunk in response.iter_content(
                        chunk_size=active_settings.download_chunk_size
                    ):
                        if chunk:
                            pdf_file.write(chunk)
            temporary_path.replace(local_path)
        paper_paths.append(local_path)
    return paper_paths


def read_pdf_pages(filepath: Path, paper_id: str) -> list[tuple[str, dict[str, str | int]]]:
    reader = PdfReader(filepath)
    pages: list[tuple[str, dict[str, str | int]]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        pages.append(
            (
                page.extract_text() or "",
                {"paper_id": paper_id, "page": page_number},
            )
        )
    return pages


def ingest_papers(
    paper_ids: Sequence[str] = REQUIRED_ARXIV_IDS,
    settings: Settings | None = None,
) -> list[str]:
    if tuple(paper_ids) != REQUIRED_ARXIV_IDS:
        raise ValueError(f"Exactly these arXiv IDs are required: {REQUIRED_ARXIV_IDS}")

    active_settings = settings or get_settings()
    paper_paths = download_papers(paper_ids, active_settings)
    document_ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, str | int]] = []

    for paper_id, paper_path in zip(paper_ids, paper_paths, strict=True):
        for page_number, (text, metadata) in enumerate(
            read_pdf_pages(paper_path, paper_id), start=1
        ):
            document_ids.append(f"{paper_id}-page-{page_number}")
            documents.append(text)
            metadatas.append(metadata)

    store = VectorStore(active_settings.chroma_path, active_settings.collection_name)
    store.upsert(document_ids, documents, metadatas)
    return list(paper_ids)
