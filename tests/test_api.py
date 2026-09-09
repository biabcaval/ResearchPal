from unittest.mock import Mock

from fastapi.testclient import TestClient

from researchpal.api.dependencies import get_research_agent, get_vector_store
from researchpal.api.main import app
from researchpal.models import AskRequest, AskResponse, ChunkMetadata, RetrievedDocument


class FakeAgent:
    def ask(self, request: AskRequest) -> AskResponse:
        return AskResponse(
            answer=f"Echo: {request.question}",
            sources=[],
            evidence_found=False,
        )


def test_health_endpoint_returns_ok_status() -> None:
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_endpoint_uses_injected_agent_without_external_calls() -> None:
    app.dependency_overrides[get_research_agent] = lambda: FakeAgent()
    try:
        response = TestClient(app).post("/ask", json={"question": "O que é atenção?"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "question": "O que é atenção?",
        "answer": "Echo: O que é atenção?",
    }


def test_ask_endpoint_rejects_extra_fields() -> None:
    response = TestClient(app).post(
        "/ask",
        json={"question": "pergunta", "limit": 5},
    )

    assert response.status_code == 422


def test_query_endpoint_uses_search_documents() -> None:
    store = Mock()
    store.search.return_value = [
        RetrievedDocument(
            identifier="1706.03762-page-1-chunk-0",
            text="attention is all you need",
            metadata=ChunkMetadata(paper_id="1706.03762", page=1),
            distance=0.1,
        )
    ]
    app.dependency_overrides[get_vector_store] = lambda: store
    try:
        response = TestClient(app).post(
            "/query",
            json={"query": "attention", "limit": 3},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"][0]["identifier"] == "1706.03762-page-1-chunk-0"
    store.search.assert_called_once_with("attention", 3)
