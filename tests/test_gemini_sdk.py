from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from researchpal.agent.gemini_sdk import (
    GeminiFunctionCall,
    first_candidate,
    iter_function_calls,
    parse_gemini_function_call,
)


def test_parse_gemini_function_call_maps_sdk_args_into_pydantic_model() -> None:
    sdk_call = SimpleNamespace(
        name="search_documents",
        args={"query": "attention", "limit": 5},
    )

    parsed = parse_gemini_function_call(sdk_call)

    assert parsed == GeminiFunctionCall(
        name="search_documents",
        arguments={"query": "attention", "limit": 5},
    )


def test_parse_gemini_function_call_treats_missing_args_as_empty_mapping() -> None:
    parsed = parse_gemini_function_call(SimpleNamespace(name="extract_section", args=None))

    assert parsed.arguments == {}


def test_parse_gemini_function_call_rejects_non_mapping_args() -> None:
    with pytest.raises(ValidationError):
        parse_gemini_function_call(SimpleNamespace(name="search_documents", args=["query"]))


def test_iter_function_calls_yields_pydantic_models_and_skips_plain_parts() -> None:
    parts = [
        SimpleNamespace(function_call=None),
        SimpleNamespace(text="ignore me"),
        SimpleNamespace(
            function_call=SimpleNamespace(
                name="extract_section",
                args={"paper_id": "1706.03762", "section": "abstract"},
            )
        ),
    ]

    parsed = list(iter_function_calls(parts))

    assert parsed == [
        GeminiFunctionCall(
            name="extract_section",
            arguments={"paper_id": "1706.03762", "section": "abstract"},
        )
    ]


def test_first_candidate_returns_none_when_response_has_no_candidates() -> None:
    assert first_candidate(SimpleNamespace(candidates=[], text="")) is None
    assert first_candidate(SimpleNamespace(text="no candidates attr")) is None
