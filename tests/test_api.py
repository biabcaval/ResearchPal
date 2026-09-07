from fastapi.testclient import TestClient

from researchpal.api.dependencies import get_research_agent
from researchpal.api.main import app


class FakeAgent:
    def ask(self, request):
        return type("AgentResponse", (), {"answer": f"Echo: {request.question}"})()


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
