from __future__ import annotations

import logging
from typing import Protocol

from langchain.messages import HumanMessage, SystemMessage
from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

from researchpal.agent.errors import GEMINI_UPSTREAM_ERRORS, ModelUnavailableError
from researchpal.models import AnswerSanityCheck

logger = logging.getLogger(__name__)

SANITY_FAIL_ERROR = "Answer did not address the question"
SANITY_REWRITE_ERROR = "Answer rewritten after sanity check"
SANITY_REJECT_ERROR = "Answer rejected after sanity check retry"
SANITY_FALLBACK_ANSWER = (
    "Não consegui responder de forma completa à pergunta com as "
    "evidências disponíveis nos artigos indexados."
)
JUDGE_INSTRUCTION = """
You check whether an assistant answer addresses the user's question.
Set addresses_question true only if the answer is on-topic AND covers
every asked-for part of the question.
Honest statements that the indexed papers do not contain enough evidence
pass when they directly respond to the question.
A detailed summary that ignores a requested comparison or other asked-for
part fails.
Write unanswered_parts in the same language as the user question.
Leave unanswered_parts empty when addresses_question is true.
reason is a short English phrase for logs only.
"""


class AnswerSanityChecker(Protocol):
    """Decides if an answer addresses the user question."""

    def check(self, question: str, answer: str) -> AnswerSanityCheck: ...


class StructuredOutputModel(Protocol):
    """Chat model that can emit `AnswerSanityCheck`."""

    def with_structured_output(
        self, schema: type[AnswerSanityCheck]
    ) -> StructuredInvoke: ...


class StructuredInvoke(Protocol):
    def invoke(self, messages: object) -> object: ...


class GeminiAnswerSanityChecker:
    """Structured Gemini judge: question + answer only, no tools."""

    def __init__(self, model: StructuredOutputModel) -> None:
        self._structured = model.with_structured_output(AnswerSanityCheck)

    def check(self, question: str, answer: str) -> AnswerSanityCheck:
        try:
            raw = self._structured.invoke(
                [
                    SystemMessage(content=JUDGE_INSTRUCTION),
                    HumanMessage(
                        content=f"Question:\n{question}\n\nAnswer:\n{answer}"
                    ),
                ]
            )
            result = _coerce_check(raw)
        except GEMINI_UPSTREAM_ERRORS as error:
            logger.warning("Sanity judge Gemini request failed: %s", error)
            raise ModelUnavailableError(
                f"Sanity check Gemini request failed: {error}"
            ) from error
        except (
            ValidationError,
            TypeError,
            ValueError,
            OutputParserException,
        ) as error:
            logger.warning("Invalid sanity check output: %s", error)
            return _invalid_check()

        if result.addresses_question:
            return result.model_copy(update={"unanswered_parts": []})
        return result


def _coerce_check(raw: object) -> AnswerSanityCheck:
    if isinstance(raw, AnswerSanityCheck):
        return raw
    if isinstance(raw, dict):
        return AnswerSanityCheck.model_validate(raw)
    raise TypeError(f"Unexpected sanity check payload: {type(raw)!r}")


def _invalid_check() -> AnswerSanityCheck:
    return AnswerSanityCheck(
        addresses_question=False,
        unanswered_parts=[],
        reason="invalid judge output",
    )
