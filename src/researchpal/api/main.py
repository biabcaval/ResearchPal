import logging
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status

from researchpal.agent import GeminiResearchAgent, ModelUnavailableError
from researchpal.api.dependencies import get_research_agent, get_vector_store
from researchpal.api.schemas import AskHttpResponse, HealthResponse
from researchpal.models import (
    AskRequest,
    RetrievedDocument,
    SearchToolParams,
    ToolResult,
)
from researchpal.tools import VectorStore, search_documents

app = FastAPI(title="ResearchPal")
logger = logging.getLogger(__name__)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/query")
def query(
    params: SearchToolParams,
    store: Annotated[VectorStore, Depends(get_vector_store)],
) -> ToolResult[list[RetrievedDocument]]:
    """Search indexed chunks using the same `search_documents` tool path."""
    return search_documents(params=params, store=store)


@app.post("/ask", response_model=AskHttpResponse)
def ask(
    request: AskRequest,
    agent: Annotated[GeminiResearchAgent, Depends(get_research_agent)],
) -> AskHttpResponse:
    try:
        result = agent.ask(request)
    except ModelUnavailableError as error:
        # Upstream quota/outage must not be reported as missing evidence.
        logger.warning("Ask request failed because Gemini is unavailable: %s", error)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error
    return AskHttpResponse(question=request.question, answer=result.answer)
