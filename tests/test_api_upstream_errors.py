import logging

import pytest
from fastapi.testclient import TestClient

from researchpal.agent import ModelUnavailableError
from researchpal.api.dependencies import get_research_agent
from researchpal.api.main import app
from researchpal.models import AskRequest, AskResponse


class QuotaExhaustedAgent:
    def ask(self, request: AskRequest) -> AskResponse:
        raise ModelUnavailableError(
            "Gemini request failed: 429 RESOURCE_EXHAUSTED. You exceeded your current quota"
        )


def test_ask_returns_503_with_the_upstream_reason(caplog: pytest.LogCaptureFixture) -> None:
    app.dependency_overrides[get_research_agent] = lambda: QuotaExhaustedAgent()
    try:
        with caplog.at_level(logging.WARNING, logger="researchpal.api.main"):
            response = TestClient(app).post("/ask", json={"question": "Qual a conclusão?"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert "RESOURCE_EXHAUSTED" in response.json()["detail"]
    assert "Gemini is unavailable" in caplog.text
