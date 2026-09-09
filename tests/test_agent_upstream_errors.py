from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from google.genai import errors as genai_errors

from researchpal.agent import ModelUnavailableError, gemini_agent
from researchpal.agent.gemini_agent import GeminiResearchAgent
from researchpal.config import Settings
from researchpal.models import RetrievedDocument

QUOTA_ERROR = genai_errors.APIError(
    429,
    {"error": {"message": "You exceeded your current quota", "status": "RESOURCE_EXHAUSTED"}},
)


class FakeStore:
    def search(self, question: str, limit: int) -> list[RetrievedDocument]:
        return [
            RetrievedDocument(
                identifier="1706.03762-page-1-chunk-0",
                text="Attention improves sequence modeling.",
                metadata={"paper_id": "1706.03762", "page": 1},
                distance=0.2,
            )
        ]


def _agent(monkeypatch: pytest.MonkeyPatch, client_factory: type[Any]) -> GeminiResearchAgent:
    settings = Settings(gemini_api_key="test-key", chroma_path=Path("data/chroma"))
    monkeypatch.setattr(gemini_agent.genai, "Client", client_factory)
    monkeypatch.setattr(gemini_agent, "VectorStore", lambda *args: FakeStore())
    return GeminiResearchAgent(settings)


def test_ask_raises_model_unavailable_when_gemini_rejects_the_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingClient:
        def __init__(self, **kwargs: Any) -> None:
            self.models = self

        def generate_content(self, **kwargs: Any) -> None:
            raise QUOTA_ERROR

    agent = _agent(monkeypatch, FailingClient)

    with pytest.raises(ModelUnavailableError, match="quota"):
        agent.ask(gemini_agent.AskRequest(question="Qual a conclusão?"))


def test_ask_raises_model_unavailable_when_final_synthesis_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A quota failure after tool rounds must not be reported as missing evidence."""

    class ToolLoopingClient:
        def __init__(self, **kwargs: Any) -> None:
            self.calls = 0
            self.models = self

        def generate_content(self, **kwargs: Any) -> SimpleNamespace:
            self.calls += 1
            if self.calls > gemini_agent.MAX_TOOL_ROUNDS:
                raise QUOTA_ERROR
            function_call = SimpleNamespace(
                name="search_documents",
                args={"query": "attention", "limit": 5},
            )
            candidate = SimpleNamespace(
                content=SimpleNamespace(parts=[SimpleNamespace(function_call=function_call)])
            )
            return SimpleNamespace(candidates=[candidate], text="")

    agent = _agent(monkeypatch, ToolLoopingClient)

    with pytest.raises(ModelUnavailableError, match="quota"):
        agent.ask(gemini_agent.AskRequest(question="Qual a conclusão?"))


def test_ask_still_reports_missing_evidence_when_tools_return_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Genuine lack of evidence keeps the existing answer; only upstream failures raise."""

    class EmptyAnswerClient:
        def __init__(self, **kwargs: Any) -> None:
            self.models = self

        def generate_content(self, **kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(
                candidates=[SimpleNamespace(content=SimpleNamespace(parts=[]))],
                text="",
            )

    agent = _agent(monkeypatch, EmptyAnswerClient)

    response = agent.ask(gemini_agent.AskRequest(question="Pergunta fora do corpus"))

    assert response.evidence_found is False
    assert "Não encontrei evidência suficiente" in response.answer
