import json
from pathlib import Path
from typing import Protocol, get_args, get_type_hints, is_typeddict

from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages import BaseMessage

from researchpal.agent.gemini_agent import (
    AgentGraph,
    GeminiResearchAgent,
    LangChainAgentState,
    SynthesisModel,
)
from researchpal.agent.sanity import AnswerSanityChecker
from researchpal.config import Settings
from researchpal.models import (
    AnswerSanityCheck,
    AskRequest,
    RetrievedDocument,
    ToolResult,
)


class AlwaysPassChecker:
    def check(self, question: str, answer: str) -> AnswerSanityCheck:
        return AnswerSanityCheck(
            addresses_question=True, unanswered_parts=[], reason="ok"
        )


def test_constructor_types_graph_and_synthesis_as_protocols() -> None:
    hints = get_type_hints(GeminiResearchAgent.__init__)
    graph_args = set(get_args(hints["graph"]))
    synthesis_args = set(get_args(hints["synthesis_model"]))

    assert issubclass(AgentGraph, Protocol)
    assert issubclass(SynthesisModel, Protocol)
    assert AgentGraph in graph_args
    assert type(None) in graph_args
    assert SynthesisModel in synthesis_args
    assert type(None) in synthesis_args
    sanity_args = set(get_args(hints["sanity_checker"]))
    assert AnswerSanityChecker in sanity_args
    assert type(None) in sanity_args


def test_agent_graph_invoke_uses_langchain_agent_state() -> None:
    assert is_typeddict(LangChainAgentState)
    assert get_type_hints(LangChainAgentState)["messages"] == list[BaseMessage]

    invoke_hints = get_type_hints(AgentGraph.invoke)
    assert invoke_hints["payload"] is LangChainAgentState
    assert invoke_hints["return"] is LangChainAgentState


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
    def invoke(self, payload: LangChainAgentState) -> LangChainAgentState:
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
        synthesis_model=UnusedSynthesis(),
        sanity_checker=AlwaysPassChecker(),
    )

    response = agent.ask(AskRequest(question="Como funciona?"))

    assert response.answer == "Resposta baseada na evidência."
    assert response.evidence_found is True
    assert response.sources[0].identifier == "1706.03762-page-1-chunk-0"


class UnusedSynthesis:
    def invoke(self, messages: object) -> AIMessage:
        raise AssertionError("synthesis should not run")


def _agent_with_graph(graph: AgentGraph) -> GeminiResearchAgent:
    return GeminiResearchAgent(
        Settings(gemini_api_key="test-key", chroma_path=Path("data/chroma")),
        store=FakeStore(),
        graph=graph,
        synthesis_model=UnusedSynthesis(),
        sanity_checker=AlwaysPassChecker(),
    )


def test_ask_records_malformed_search_documents_instead_of_raising() -> None:
    class MalformedSearchGraph:
        def invoke(self, payload: LangChainAgentState) -> LangChainAgentState:
            return {
                "messages": [
                    HumanMessage(content="Como funciona?"),
                    ToolMessage(
                        content=json.dumps(
                            {
                                "success": True,
                                "data": [{"identifier": "incomplete"}],
                                "error": None,
                            }
                        ),
                        name="search_documents",
                        tool_call_id="call-1",
                    ),
                    AIMessage(content="Resposta mesmo com evidência inválida."),
                ]
            }

    response = _agent_with_graph(MalformedSearchGraph()).ask(
        AskRequest(question="Como funciona?"),
    )

    assert response.answer == "Resposta mesmo com evidência inválida."
    assert response.sources == []
    assert response.evidence_found is False
    assert any("Invalid tool result" in error for error in response.tool_errors)


def test_ask_discards_the_whole_search_batch_when_one_document_is_malformed() -> None:
    valid_document = {
        "identifier": "1706.03762-page-1-chunk-0",
        "text": "Attention improves sequence modeling.",
        "metadata": {"paper_id": "1706.03762", "page": 1},
        "distance": 0.2,
    }

    class MixedSearchGraph:
        def invoke(self, payload: LangChainAgentState) -> LangChainAgentState:
            return {
                "messages": [
                    HumanMessage(content="Como funciona?"),
                    ToolMessage(
                        content=json.dumps(
                            {
                                "success": True,
                                "data": [valid_document, {"identifier": "incomplete"}],
                                "error": None,
                            }
                        ),
                        name="search_documents",
                        tool_call_id="call-1",
                    ),
                    AIMessage(content="Resposta sem evidência parcial."),
                ]
            }

    response = _agent_with_graph(MixedSearchGraph()).ask(
        AskRequest(question="Como funciona?"),
    )

    assert response.answer == "Resposta sem evidência parcial."
    assert response.sources == []
    assert response.evidence_found is False
    assert any("Invalid tool result" in error for error in response.tool_errors)


def test_ask_records_malformed_extract_section_instead_of_raising() -> None:
    class MalformedSectionGraph:
        def invoke(self, payload: LangChainAgentState) -> LangChainAgentState:
            return {
                "messages": [
                    HumanMessage(content="Qual o abstract?"),
                    ToolMessage(
                        content=json.dumps(
                            {
                                "success": True,
                                "data": {"paper_id": "1706.03762"},
                                "error": None,
                            }
                        ),
                        name="extract_section",
                        tool_call_id="call-2",
                    ),
                    AIMessage(content="Não consegui extrair a seção."),
                ]
            }

    response = _agent_with_graph(MalformedSectionGraph()).ask(
        AskRequest(question="Qual o abstract?"),
    )

    assert response.answer == "Não consegui extrair a seção."
    assert response.sections == []
    assert response.evidence_found is False
    assert any("Invalid tool result" in error for error in response.tool_errors)
