import logging

import chromadb
from fastapi import HTTPException, status

from researchpal.agent import GeminiResearchAgent
from researchpal.config import get_settings
from researchpal.tools import VectorStore

logger = logging.getLogger(__name__)


def get_research_agent() -> GeminiResearchAgent:
    """Create the agent lazily so importing the ASGI app has no side effects."""
    try:
        return GeminiResearchAgent()
    except (OSError, RuntimeError, ValueError, chromadb.errors.ChromaError) as error:
        logger.warning("Failed to create the research agent: %s", error)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error


def get_vector_store() -> VectorStore:
    """Create the vector store lazily so importing the ASGI app has no side effects."""
    try:
        settings = get_settings()
        return VectorStore(settings.chroma_path, settings.collection_name)
    except (OSError, RuntimeError, ValueError, chromadb.errors.ChromaError) as error:
        logger.warning("Failed to create the vector store: %s", error)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error
