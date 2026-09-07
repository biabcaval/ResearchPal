from fastapi import FastAPI

from researchpal.agent import GeminiResearchAgent, ResearchAgent
from researchpal.config import get_settings
from researchpal.models import (
    AskRequest,
    AskResponse,
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


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        return GeminiResearchAgent().ask(request)
    except (RuntimeError, ValueError) as error:
        return AskResponse(
            answer="Não foi possível consultar o agente de pesquisa.",
            sources=[],
            sections=[],
            evidence_found=False,
            tool_errors=[str(error)],
        )
