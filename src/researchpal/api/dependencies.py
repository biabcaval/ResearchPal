import chromadb
from fastapi import HTTPException, status

from researchpal.agent import GeminiResearchAgent


def get_research_agent() -> GeminiResearchAgent:
    """Create the agent lazily so importing the ASGI app has no side effects."""
    try:
        return GeminiResearchAgent()
    except (OSError, RuntimeError, ValueError, chromadb.errors.ChromaError) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error
