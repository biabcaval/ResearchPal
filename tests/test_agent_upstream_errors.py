from pathlib import Path

import pytest
from google.genai import errors as genai_errors
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain_google_genai.chat_models import GoogleRateLimitError

from researchpal.agent import ModelUnavailableError
from researchpal.agent.gemini_agent import GeminiResearchAgent
from researchpal.config import Settings
from researchpal.models import AskRequest, RetrievedDocument, ToolResult

QUOTA_ERROR = genai_errors.APIError(
    429,
    {
        "error": {
            "message": "You exceeded your current quota",
            "status": "RESOURCE_EXHAUSTED",
        }
    },
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


def _agent(
    *,
    graph: object,
    synthesis_model: object | None = None,
) -> GeminiResearchAgent:
    settings = Settings(gemini_api_key="test-key", chroma_path=Path("data/chroma"))
    return GeminiResearchAgent(
        settings,
        store=FakeStore(),
        graph=graph,
        synthesis_model=synthesis_model or object(),
    )


def test_ask_raises_model_unavailable_when_langchain_wraps_quota_error() -> None:
    class FailingGraph:
        def invoke(self, payload: dict[str, object]) -> dict[str, object]:
            raise GoogleRateLimitError("You exceeded your current quota")

    agent = _agent(graph=FailingGraph())

    with pytest.raises(ModelUnavailableError, match="quota"):
        agent.ask(AskRequest(question="Qual a conclusão?"))


def test_ask_raises_model_unavailable_when_gemini_rejects_the_request() -> None:
    class FailingGraph:
        def invoke(self, payload: dict[str, object]) -> dict[str, object]:
            raise QUOTA_ERROR

    agent = _agent(graph=FailingGraph())

    with pytest.raises(ModelUnavailableError, match="quota"):
        agent.ask(AskRequest(question="Qual a conclusão?"))


def test_ask_raises_model_unavailable_when_final_synthesis_fails() -> None:
    document = RetrievedDocument(
        identifier="1706.03762-page-1-chunk-0",
        text="Attention improves sequence modeling.",
        metadata={"paper_id": "1706.03762", "page": 1},
        distance=0.2,
    )

    class LimitGraph:
        def invoke(self, payload: dict[str, object]) -> dict[str, object]:
            return {
                "messages": [
                    HumanMessage(content="Qual a conclusão?"),
                    ToolMessage(
                        content=ToolResult(
                            success=True,
                            data=[document],
                        ).model_dump_json(),
                        name="search_documents",
                        tool_call_id="call-1",
                    ),
                    AIMessage(
                        content="Model call limits exceeded: run limit (3/3)"
                    ),
                ]
            }

    class FailingSynthesis:
        def invoke(self, messages: object) -> AIMessage:
            raise GoogleRateLimitError("You exceeded your current quota")

    agent = _agent(graph=LimitGraph(), synthesis_model=FailingSynthesis())

    with pytest.raises(ModelUnavailableError, match="quota"):
        agent.ask(AskRequest(question="Qual a conclusão?"))


def test_ask_synthesizes_without_tools_after_model_call_limit() -> None:
    document = RetrievedDocument(
        identifier="1706.03762-page-1-chunk-0",
        text="Attention improves sequence modeling.",
        metadata={"paper_id": "1706.03762", "page": 1},
        distance=0.2,
    )

    class LimitGraph:
        def invoke(self, payload: dict[str, object]) -> dict[str, object]:
            return {
                "messages": [
                    HumanMessage(content="Qual a conclusão?"),
                    ToolMessage(
                        content=ToolResult(
                            success=True,
                            data=[document],
                        ).model_dump_json(),
                        name="search_documents",
                        tool_call_id="call-1",
                    ),
                    AIMessage(
                        content="Model call limits exceeded: run limit (3/3)"
                    ),
                ]
            }

    class FakeSynthesis:
        def invoke(self, messages: object) -> AIMessage:
            return AIMessage(content="Síntese forçada.")

    agent = _agent(graph=LimitGraph(), synthesis_model=FakeSynthesis())
    response = agent.ask(AskRequest(question="Qual a conclusão?"))

    assert response.answer == "Síntese forçada."
    assert response.evidence_found is True
    assert "Maximum tool-calling rounds exceeded" in response.tool_errors
    assert "Model call limits exceeded" not in response.answer


def test_ask_still_reports_missing_evidence_when_tools_return_nothing() -> None:
    class EmptyGraph:
        def invoke(self, payload: dict[str, object]) -> dict[str, object]:
            return {
                "messages": [
                    HumanMessage(content="Pergunta fora do corpus"),
                    AIMessage(content=""),
                ]
            }

    agent = _agent(graph=EmptyGraph())
    response = agent.ask(AskRequest(question="Pergunta fora do corpus"))

    assert response.evidence_found is False
    assert "Não encontrei evidência suficiente" in response.answer
