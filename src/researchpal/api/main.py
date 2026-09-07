from typing import Annotated

from fastapi import Depends, FastAPI

from researchpal.agent import GeminiResearchAgent, ResearchAgent
from researchpal.config import get_settings
from researchpal.api.dependencies import get_research_agent
from researchpal.api.schemas import AskHttpRequest, AskHttpResponse
from researchpal.models import (
    AskRequest,
    QueryRequest,
    QueryResponse,
)

app = FastAPI(title="ResearchPal")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query")
def query(request: QueryRequest) -> QueryResponse:
    settings = get_settings()
    return ResearchAgent(settings).store.query(request.question, request.limit)


@app.post("/ask", response_model=AskHttpResponse)
def ask(
    request: AskHttpRequest,
    agent: Annotated[GeminiResearchAgent, Depends(get_research_agent)],
) -> AskHttpResponse:
    result = agent.ask(AskRequest(question=request.question))
    return AskHttpResponse(question=request.question, answer=result.answer)
