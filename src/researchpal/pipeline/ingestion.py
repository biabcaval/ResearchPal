import logging
from collections.abc import Sequence
from pathlib import Path

import requests

from researchpal.config import REQUIRED_ARXIV_IDS, Settings, get_settings
from researchpal.models import ChunkMetadata
from researchpal.tools import VectorStore
from researchpal.tools.pdf import read_pdf_pages

logger = logging.getLogger(__name__)


def arxiv_pdf_url(paper_id: str) -> str:
    return f"https://arxiv.org/pdf/{paper_id}.pdf"


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Split text into deterministic character chunks with a fixed overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    normalized_text = text.strip()
    if not normalized_text:
        return []

    step = chunk_size - overlap
    return [
        normalized_text[start : start + chunk_size]
        for start in range(0, len(normalized_text), step)
    ]


def download_papers(
    paper_ids: Sequence[str] = REQUIRED_ARXIV_IDS,
    settings: Settings | None = None,
) -> list[Path]:
    active_settings = settings or get_settings()
    active_settings.pdf_directory.mkdir(parents=True, exist_ok=True)
    paper_paths: list[Path] = []

    for paper_id in paper_ids:
        local_path = active_settings.pdf_directory / f"{paper_id}.pdf"
        if not local_path.is_file() or local_path.stat().st_size == 0:
            temporary_path = local_path.with_suffix(".pdf.part")
            try:
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
                if temporary_path.stat().st_size == 0:
                    raise ValueError(f"Downloaded an empty PDF for {paper_id}")
                temporary_path.replace(local_path)
            except (OSError, requests.RequestException, ValueError) as error:
                temporary_path.unlink(missing_ok=True)
                logger.warning("Failed to download PDF for %s: %s", paper_id, error)
                raise RuntimeError(f"Failed to download PDF for {paper_id}") from error
        paper_paths.append(local_path)
    return paper_paths


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
    metadatas: list[ChunkMetadata] = []

    for paper_id, paper_path in zip(paper_ids, paper_paths, strict=True):
        pages = read_pdf_pages(paper_path, paper_id)
        if not any(text.strip() for text, _ in pages):
            raise ValueError(f"PDF contains no extractable text: {paper_id}")

        for page_number, (text, page_metadata) in enumerate(pages, start=1):
            chunks = chunk_text(
                text,
                chunk_size=active_settings.chunk_size,
                overlap=active_settings.chunk_overlap,
            )
            for chunk_index, chunk in enumerate(chunks):
                document_ids.append(f"{paper_id}-page-{page_number}-chunk-{chunk_index}")
                documents.append(chunk)
                metadatas.append(
                    ChunkMetadata(
                        paper_id=page_metadata.paper_id,
                        page=page_metadata.page,
                        chunk_index=chunk_index,
                        chunk_size=active_settings.chunk_size,
                        chunk_overlap=active_settings.chunk_overlap,
                    )
                )

    if not documents:
        raise ValueError("No non-empty document chunks were produced")

    store = VectorStore(active_settings.chroma_path, active_settings.collection_name)
    store.upsert(document_ids, documents, metadatas)
    return list(paper_ids)
