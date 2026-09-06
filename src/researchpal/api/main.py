from fastapi import FastAPI

from researchpal.agent import ResearchAgent
from researchpal.models import QueryRequest

app = FastAPI(title="ResearchPal")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query")
def query(request: QueryRequest) -> dict[str, object]:
    result = ResearchAgent().search(request.question, request.limit)
    return {"ids": result.ids, "distances": result.distances}
