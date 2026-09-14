import logging
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from researchpal.api.dependencies import get_research_agent, get_vector_store
from researchpal.api.main import app
from researchpal.models import (
    AskRequest,
    AskResponse,
    ChunkMetadata,
    Citation,
    RetrievedDocument,
)


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
        "citations": [],
    }


def test_ask_endpoint_includes_citations_from_the_agent() -> None:
    class CitedAgent:
        def ask(self, request: AskRequest) -> AskResponse:
            return AskResponse(
                answer=f"Echo: {request.question} [1]",
                sources=[],
                evidence_found=True,
                citations=[
                    Citation(
                        number=1,
                        identifier="1706.03762-page-1-chunk-0",
                        paper_id="1706.03762",
                        page=1,
                        snippet="Attention improves sequence modeling.",
                    )
                ],
            )

    app.dependency_overrides[get_research_agent] = lambda: CitedAgent()
    try:
        response = TestClient(app).post("/ask", json={"question": "O que é atenção?"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["citations"] == [
        {
            "number": 1,
            "identifier": "1706.03762-page-1-chunk-0",
            "paper_id": "1706.03762",
            "page": 1,
            "snippet": "Attention improves sequence modeling.",
        }
    ]


def test_ask_endpoint_rejects_extra_fields() -> None:
    app.dependency_overrides[get_research_agent] = lambda: FakeAgent()
    try:
        response = TestClient(app).post(
            "/ask",
            json={"question": "pergunta", "limit": 5},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422


def test_ask_endpoint_rejects_question_over_50_words() -> None:
    app.dependency_overrides[get_research_agent] = lambda: FakeAgent()
    try:
        response = TestClient(app).post(
            "/ask",
            json={"question": " ".join(f"word-{index}" for index in range(51))},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422


def test_ask_endpoint_accepts_question_with_exactly_50_words() -> None:
    question = " ".join(f"word-{index}" for index in range(50))
    app.dependency_overrides[get_research_agent] = lambda: FakeAgent()
    try:
        response = TestClient(app).post("/ask", json={"question": question})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["question"] == question


def test_ask_endpoint_rejects_whitespace_only_question() -> None:
    app.dependency_overrides[get_research_agent] = lambda: FakeAgent()
    try:
        response = TestClient(app).post("/ask", json={"question": " \t\n "})
    finally:
        app.dependency_overrides.clear()

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


def test_query_endpoint_rejects_query_over_50_words() -> None:
    store = Mock()
    store.search.return_value = []
    app.dependency_overrides[get_vector_store] = lambda: store
    try:
        response = TestClient(app).post(
            "/query",
            json={"query": " ".join(f"word-{index}" for index in range(51))},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422


def test_ask_agent_construction_503_hides_local_path(
    caplog: pytest.LogCaptureFixture,
) -> None:
    local_path = "/Users/private/researchpal/chroma.sqlite3"
    with (
        patch(
            "researchpal.api.dependencies.GeminiResearchAgent",
            side_effect=OSError(f"Cannot open {local_path}"),
        ),
        caplog.at_level(logging.WARNING, logger="researchpal.api.dependencies"),
    ):
        response = TestClient(app).post("/ask", json={"question": "What happened?"})

    assert response.status_code == 503
    assert response.json()["detail"] == "Research service is temporarily unavailable."
    assert local_path not in response.text
    assert local_path in caplog.text


def test_query_store_construction_503_hides_local_path(
    caplog: pytest.LogCaptureFixture,
) -> None:
    local_path = "/Users/private/researchpal/chroma.sqlite3"
    with (
        patch(
            "researchpal.api.dependencies.VectorStore",
            side_effect=OSError(f"Cannot open {local_path}"),
        ),
        caplog.at_level(logging.WARNING, logger="researchpal.api.dependencies"),
    ):
        response = TestClient(app).post(
            "/query",
            json={"query": "attention", "limit": 3},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "Research service is temporarily unavailable."
    assert local_path not in response.text
    assert local_path in caplog.text
