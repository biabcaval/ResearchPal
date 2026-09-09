from pathlib import Path
from types import SimpleNamespace
from typing import Any

from researchpal.agent import gemini_agent
from researchpal.agent.gemini_agent import GeminiResearchAgent
from researchpal.config import Settings
from researchpal.models import RetrievedDocument


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


class FakeClient:
    def __init__(self, **kwargs: Any) -> None:
        self.calls = 0
        self.models = self

    def generate_content(self, **kwargs: Any) -> SimpleNamespace:
        self.calls += 1
        if self.calls == 1:
            function_call = SimpleNamespace(
                name="search_documents",
                args={"query": "attention", "limit": 5},
            )
            candidate = SimpleNamespace(
                content=SimpleNamespace(
                    parts=[SimpleNamespace(function_call=function_call)]
                )
            )
            return SimpleNamespace(candidates=[candidate], text="")
        return SimpleNamespace(
            candidates=[SimpleNamespace(content=SimpleNamespace(parts=[]))],
            text="Resposta baseada na evidência.",
        )


def test_agent_executes_function_call_and_synthesizes_answer(monkeypatch) -> None:
    settings = Settings(
        gemini_api_key="test-key",
        chroma_path=Path("data/chroma"),
    )
    monkeypatch.setattr(gemini_agent.genai, "Client", FakeClient)
    monkeypatch.setattr(gemini_agent, "VectorStore", lambda *args: FakeStore())

    agent = GeminiResearchAgent(settings)
    response = agent.ask(gemini_agent.AskRequest(question="Como funciona?"))

    assert response.answer == "Resposta baseada na evidência."
    assert response.evidence_found is True
    assert response.sources[0].identifier == "1706.03762-page-1-chunk-0"
