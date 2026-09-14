from pathlib import Path

import pytest
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain_google_genai.chat_models import GoogleRateLimitError

from researchpal.agent.errors import ModelUnavailableError
from researchpal.agent.gemini_agent import (
    CANNED_ANSWERS,
    GeminiResearchAgent,
    LangChainAgentState,
    SynthesisModel,
)
from researchpal.agent.sanity import (
    SANITY_FAIL_ERROR,
    SANITY_FALLBACK_ANSWER,
    SANITY_REJECT_ERROR,
    SANITY_REWRITE_ERROR,
)
from researchpal.config import Settings
from researchpal.models import (
    AnswerSanityCheck,
    AskRequest,
    RetrievedDocument,
    ToolResult,
)


class FakeStore:
    def search(self, question: str, limit: int) -> list[RetrievedDocument]:
        return []


class RecordingSynthesis:
    def __init__(self, text: str = "Resposta reescrita completa.") -> None:
        self.text = text
        self.calls: list[object] = []

    def invoke(self, messages: object) -> AIMessage:
        self.calls.append(messages)
        return AIMessage(content=self.text)


class SequenceChecker:
    def __init__(self, results: list[AnswerSanityCheck | BaseException]) -> None:
        self._results = list(results)
        self.calls: list[tuple[str, str]] = []

    def check(self, question: str, answer: str) -> AnswerSanityCheck:
        self.calls.append((question, answer))
        item = self._results.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _graph_with_answer(answer: str) -> object:
    document = RetrievedDocument(
        identifier="1706.03762-page-1-chunk-0",
        text="Attention improves sequence modeling.",
        metadata={"paper_id": "1706.03762", "page": 1},
        distance=0.2,
    )

    class Graph:
        def invoke(self, payload: LangChainAgentState) -> LangChainAgentState:
            return {
                "messages": [
                    HumanMessage(content=payload["messages"][0].content),
                    ToolMessage(
                        content=ToolResult(
                            success=True, data=[document]
                        ).model_dump_json(),
                        name="search_documents",
                        tool_call_id="call-1",
                    ),
                    AIMessage(content=answer),
                ]
            }

    return Graph()


def _agent(
    *,
    answer: str = "Resposta baseada na evidência.",
    checker: SequenceChecker,
    synthesis: SynthesisModel | None = None,
) -> tuple[GeminiResearchAgent, SynthesisModel]:
    synth = synthesis or RecordingSynthesis()
    agent = GeminiResearchAgent(
        Settings(gemini_api_key="test-key", chroma_path=Path("data/chroma")),
        store=FakeStore(),
        graph=_graph_with_answer(answer),
        synthesis_model=synth,
        sanity_checker=checker,
    )
    return agent, synth


def test_ask_returns_answer_unchanged_when_sanity_passes() -> None:
    checker = SequenceChecker(
        [
            AnswerSanityCheck(
                addresses_question=True, unanswered_parts=[], reason="ok"
            )
        ]
    )
    agent, synth = _agent(checker=checker)

    response = agent.ask(AskRequest(question="Como funciona?"))

    assert response.answer == "Resposta baseada na evidência."
    assert synth.calls == []
    assert SANITY_FAIL_ERROR not in response.tool_errors
    assert checker.calls == [("Como funciona?", "Resposta baseada na evidência.")]


def test_ask_rewrites_once_when_first_check_fails() -> None:
    checker = SequenceChecker(
        [
            AnswerSanityCheck(
                addresses_question=False,
                unanswered_parts=["a comparação pedida"],
                reason="incomplete",
            ),
            AnswerSanityCheck(
                addresses_question=True, unanswered_parts=[], reason="ok"
            ),
        ]
    )
    agent, synth = _agent(checker=checker)
    response = agent.ask(AskRequest(question="Compare os artigos"))

    assert response.answer == "Resposta reescrita completa."
    assert SANITY_FAIL_ERROR in response.tool_errors
    assert SANITY_REWRITE_ERROR in response.tool_errors
    assert SANITY_REJECT_ERROR not in response.tool_errors
    rewrite_prompt = str(synth.calls[0][-1].content)
    assert "a comparação pedida" in rewrite_prompt
    assert "Não chame mais nenhuma ferramenta" in rewrite_prompt


def test_ask_returns_fallback_when_rewrite_still_fails() -> None:
    checker = SequenceChecker(
        [
            AnswerSanityCheck(
                addresses_question=False,
                unanswered_parts=["conclusão"],
                reason="incomplete",
            ),
            AnswerSanityCheck(
                addresses_question=False,
                unanswered_parts=["conclusão"],
                reason="still incomplete",
            ),
        ]
    )
    agent, _synth = _agent(checker=checker)
    response = agent.ask(AskRequest(question="Qual a conclusão?"))

    assert response.answer == SANITY_FALLBACK_ANSWER
    assert SANITY_FAIL_ERROR in response.tool_errors
    assert SANITY_REWRITE_ERROR in response.tool_errors
    assert SANITY_REJECT_ERROR in response.tool_errors


def test_ask_returns_fallback_when_rewrite_is_empty() -> None:
    checker = SequenceChecker(
        [
            AnswerSanityCheck(
                addresses_question=False,
                unanswered_parts=["método"],
                reason="incomplete",
            )
        ]
    )
    synth = RecordingSynthesis(text="   ")
    agent, _ = _agent(checker=checker, synthesis=synth)
    response = agent.ask(AskRequest(question="Qual o método?"))

    assert response.answer == SANITY_FALLBACK_ANSWER
    assert SANITY_FAIL_ERROR in response.tool_errors
    assert SANITY_REWRITE_ERROR not in response.tool_errors
    assert SANITY_REJECT_ERROR in response.tool_errors


def test_ask_skips_checker_for_canned_no_evidence_answer() -> None:
    class EmptyGraph:
        def invoke(self, payload: LangChainAgentState) -> LangChainAgentState:
            return {
                "messages": [
                    HumanMessage(content="Pergunta fora do corpus"),
                    AIMessage(content=""),
                ]
            }

    class UnusedChecker:
        def check(self, question: str, answer: str) -> AnswerSanityCheck:
            raise AssertionError("sanity checker should not run")

    agent = GeminiResearchAgent(
        Settings(gemini_api_key="test-key", chroma_path=Path("data/chroma")),
        store=FakeStore(),
        graph=EmptyGraph(),
        synthesis_model=RecordingSynthesis(),
        sanity_checker=UnusedChecker(),
    )
    response = agent.ask(AskRequest(question="Pergunta fora do corpus"))

    assert response.answer in CANNED_ANSWERS
    assert "Não encontrei evidência suficiente" in response.answer
    assert SANITY_FAIL_ERROR not in response.tool_errors


def test_ask_passes_honest_model_insufficient_evidence_when_checker_says_so() -> None:
    honest = (
        "Não há evidência suficiente nos artigos indexados para comparar "
        "os dois métodos."
    )
    checker = SequenceChecker(
        [
            AnswerSanityCheck(
                addresses_question=True, unanswered_parts=[], reason="honest gap"
            )
        ]
    )
    agent, synth = _agent(answer=honest, checker=checker)
    response = agent.ask(AskRequest(question="Compare os métodos"))

    assert response.answer == honest
    assert synth.calls == []


def test_ask_raises_when_judge_hits_quota() -> None:
    checker = SequenceChecker(
        [GoogleRateLimitError("You exceeded your current quota")]
    )
    agent, _ = _agent(checker=checker)
    with pytest.raises(ModelUnavailableError, match="quota"):
        agent.ask(AskRequest(question="Como funciona?"))


def test_ask_raises_when_rewrite_hits_quota() -> None:
    class FailingSynthesis:
        def invoke(self, messages: object) -> AIMessage:
            raise GoogleRateLimitError("You exceeded your current quota")

    checker = SequenceChecker(
        [
            AnswerSanityCheck(
                addresses_question=False,
                unanswered_parts=["detalhe"],
                reason="incomplete",
            )
        ]
    )
    agent, _ = _agent(checker=checker, synthesis=FailingSynthesis())  # type: ignore[arg-type]
    with pytest.raises(ModelUnavailableError, match="quota"):
        agent.ask(AskRequest(question="Como funciona?"))


def test_ask_retries_when_first_judge_payload_is_invalid() -> None:
    class InvalidThenPass:
        def __init__(self) -> None:
            self.calls = 0

        def check(self, question: str, answer: str) -> AnswerSanityCheck:
            self.calls += 1
            if self.calls == 1:
                return AnswerSanityCheck(
                    addresses_question=False,
                    unanswered_parts=[],
                    reason="invalid judge output",
                )
            return AnswerSanityCheck(
                addresses_question=True, unanswered_parts=[], reason="ok"
            )

    checker = InvalidThenPass()
    agent, synth = _agent(checker=checker)  # type: ignore[arg-type]
    response = agent.ask(AskRequest(question="Como funciona?"))

    assert response.answer == "Resposta reescrita completa."
    assert synth.calls
    assert SANITY_FAIL_ERROR in response.tool_errors
