from pathlib import Path

from langchain.messages import AIMessage, HumanMessage, ToolMessage

from researchpal.agent.gemini_agent import GeminiResearchAgent
from researchpal.config import Settings
from researchpal.models import AskRequest, RetrievedDocument, ToolResult


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


class FakeGraph:
    def invoke(self, payload: dict[str, object]) -> dict[str, object]:
        document = RetrievedDocument(
            identifier="1706.03762-page-1-chunk-0",
            text="Attention improves sequence modeling.",
            metadata={"paper_id": "1706.03762", "page": 1},
            distance=0.2,
        )
        return {
            "messages": [
                HumanMessage(content="Como funciona?"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_documents",
                            "args": {"query": "attention", "limit": 5},
                            "id": "call-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                ToolMessage(
                    content=ToolResult(success=True, data=[document]).model_dump_json(),
                    name="search_documents",
                    tool_call_id="call-1",
                ),
                AIMessage(content="Resposta baseada na evidência."),
            ]
        }


def test_agent_maps_tool_messages_and_synthesizes_answer() -> None:
    settings = Settings(
        gemini_api_key="test-key",
        chroma_path=Path("data/chroma"),
    )
    agent = GeminiResearchAgent(
        settings,
        store=FakeStore(),
        graph=FakeGraph(),
        synthesis_model=object(),
    )

    response = agent.ask(AskRequest(question="Como funciona?"))

    assert response.answer == "Resposta baseada na evidência."
    assert response.evidence_found is True
    assert response.sources[0].identifier == "1706.03762-page-1-chunk-0"
