from unittest.mock import Mock

import pytest
from google.genai import errors as genai_errors
from langchain.messages import HumanMessage, SystemMessage
from langchain_google_genai.chat_models import GoogleRateLimitError
from pydantic import ValidationError

from researchpal.agent import ModelUnavailableError
from researchpal.agent.sanity import (
    SANITY_FAIL_ERROR,
    SANITY_FALLBACK_ANSWER,
    SANITY_REJECT_ERROR,
    SANITY_REWRITE_ERROR,
    GeminiAnswerSanityChecker,
)
from researchpal.models import AnswerSanityCheck


class StubStructuredModel:
    def __init__(self, result: object) -> None:
        self.result = result
        self.invoke_messages: list[object] = []

    def with_structured_output(self, schema: type[AnswerSanityCheck]) -> "StubStructuredModel":
        assert schema is AnswerSanityCheck
        return self

    def invoke(self, messages: object) -> object:
        self.invoke_messages.append(messages)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_answer_sanity_check_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        AnswerSanityCheck(
            addresses_question=True,
            unanswered_parts=[],
            reason="ok",
            extra="nope",  # type: ignore[call-arg]
        )


def test_checker_passes_on_topic_complete_answer() -> None:
    stub = StubStructuredModel(
        AnswerSanityCheck(
            addresses_question=True,
            unanswered_parts=["should be cleared"],
            reason="covers the question",
        )
    )
    checker = GeminiAnswerSanityChecker(stub)
    result = checker.check("Como funciona o attention?", "O attention pondera tokens.")

    assert result.addresses_question is True
    assert result.unanswered_parts == []
    messages = stub.invoke_messages[0]
    assert any(isinstance(item, SystemMessage) for item in messages)
    human = next(item for item in messages if isinstance(item, HumanMessage))
    assert "Como funciona o attention?" in str(human.content)
    assert "O attention pondera tokens." in str(human.content)
    assert "Attention improves" not in str(human.content)


def test_checker_returns_unanswered_parts_when_incomplete() -> None:
    stub = StubStructuredModel(
        AnswerSanityCheck(
            addresses_question=False,
            unanswered_parts=["comparação entre os dois artigos"],
            reason="only one paper",
        )
    )
    result = GeminiAnswerSanityChecker(stub).check(
        "Compare os dois artigos",
        "O primeiro artigo usa transformers.",
    )

    assert result.addresses_question is False
    assert result.unanswered_parts == ["comparação entre os dois artigos"]


def test_checker_treats_invalid_output_as_fail() -> None:
    stub = StubStructuredModel({"addresses_question": "yes"})
    result = GeminiAnswerSanityChecker(stub).check("Q?", "A.")

    assert result.addresses_question is False
    assert result.unanswered_parts == []
    assert result.reason == "invalid judge output"


def test_checker_raises_model_unavailable_on_quota() -> None:
    stub = StubStructuredModel(GoogleRateLimitError("You exceeded your current quota"))
    with pytest.raises(ModelUnavailableError, match="quota"):
        GeminiAnswerSanityChecker(stub).check("Q?", "A.")


def test_checker_raises_model_unavailable_on_genai_api_error() -> None:
    error = genai_errors.APIError(
        429,
        {
            "error": {
                "message": "You exceeded your current quota",
                "status": "RESOURCE_EXHAUSTED",
            }
        },
    )
    stub = StubStructuredModel(error)
    with pytest.raises(ModelUnavailableError, match="quota"):
        GeminiAnswerSanityChecker(stub).check("Q?", "A.")


def test_sanity_copy_constants_match_spec() -> None:
    assert SANITY_FAIL_ERROR == "Answer did not address the question"
    assert SANITY_REWRITE_ERROR == "Answer rewritten after sanity check"
    assert SANITY_REJECT_ERROR == "Answer rejected after sanity check retry"
    assert SANITY_FALLBACK_ANSWER == (
        "Não consegui responder de forma completa à pergunta com as "
        "evidências disponíveis nos artigos indexados."
    )
