"""Typed adapters for google-genai generate_content responses."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field


class GeminiFunctionCall(BaseModel):
    """Parsed Gemini function-call payload after the SDK object boundary."""

    model_config = ConfigDict(extra="forbid")

    name: str = ""
    arguments: dict[str, object] = Field(default_factory=dict)


class GeminiFunctionCallLike(Protocol):
    """SDK function-call object (`name` + `args`)."""

    name: str | None
    args: Mapping[str, object] | None


class GeminiPartLike(Protocol):
    """SDK content part that may carry a function call."""

    function_call: GeminiFunctionCallLike | None


class GeminiContentLike(Protocol):
    parts: Sequence[GeminiPartLike] | None


class GeminiCandidateLike(Protocol):
    content: GeminiContentLike


class GeminiGenerateContentResponse(Protocol):
    """Subset of `GenerateContentResponse` used by the agent."""

    text: str | None
    candidates: Sequence[GeminiCandidateLike] | None


def parse_gemini_function_call(
    function_call: GeminiFunctionCallLike,
) -> GeminiFunctionCall:
    """Validate an SDK function-call object into a typed model.

    Args:
        function_call: Gemini SDK (or test double) function-call object.

    Returns:
        Name and argument mapping ready for tool-specific Pydantic validation.
    """
    raw_args = getattr(function_call, "args", None)
    if raw_args is None:
        arguments: object = {}
    elif isinstance(raw_args, Mapping):
        arguments = dict(raw_args)
    else:
        arguments = raw_args
    return GeminiFunctionCall.model_validate(
        {
            "name": getattr(function_call, "name", None) or "",
            "arguments": arguments,
        }
    )


def iter_function_calls(parts: Iterable[object]) -> Iterable[GeminiFunctionCall]:
    """Yield parsed function calls, skipping parts that do not request a tool."""
    for part in parts:
        function_call = getattr(part, "function_call", None)
        if function_call is None:
            continue
        yield parse_gemini_function_call(function_call)


def first_candidate(
    response: GeminiGenerateContentResponse,
) -> GeminiCandidateLike | None:
    """Return the first candidate, if Gemini produced one."""
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return None
    return candidates[0]
