from fastapi import FastAPI

from researchpal.agent import ResearchAgent
from researchpal.config import get_settings
from researchpal.models import (
    AskRequest,
    AskResponse,
    QueryRequest,
    QueryResponse,
    SearchToolParams,
)

app = FastAPI(title="ResearchPal")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query")
def query(request: QueryRequest) -> QueryResponse:
    settings = get_settings()
    return ResearchAgent(settings).store.query(request.question, request.limit)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    settings = get_settings()
    params = SearchToolParams(
        query=request.question,
        limit=request.limit or settings.retrieval_limit,
    )
    sources = ResearchAgent(settings).search(params)
    return AskResponse(
        answer="Retrieval completed. No generative answer provider is configured yet.",
        sources=sources,
    )
