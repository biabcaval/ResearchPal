from types import SimpleNamespace
from typing import Any

import pytest
from google.genai import errors as genai_errors

from researchpal.config import Settings
from researchpal.tools import english_retrieval
from researchpal.tools.english_retrieval import to_english_retrieval_query


def test_to_english_retrieval_query_rewrites_portuguese_with_injected_generator() -> None:
    def fake_generate(query: str) -> str:
        assert query == "Qual é o mecanismo central proposto no paper Attention Is All You Need?"
        return "central mechanism self-attention Attention Is All You Need"

    result = to_english_retrieval_query(
        "Qual é o mecanismo central proposto no paper Attention Is All You Need?",
        generate_english=fake_generate,
    )

    assert result == "central mechanism self-attention Attention Is All You Need"


def test_to_english_retrieval_query_rewrites_english_with_injected_generator() -> None:
    def fake_generate(query: str) -> str:
        assert query == "What is attention?"
        return "attention mechanism transformers"

    result = to_english_retrieval_query(
        "What is attention?",
        generate_english=fake_generate,
    )

    assert result == "attention mechanism transformers"


def test_to_english_retrieval_query_falls_back_when_rewrite_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fail(_query: str) -> str:
        raise RuntimeError("quota exceeded")

    with caplog.at_level("WARNING", logger="researchpal.tools.english_retrieval"):
        result = to_english_retrieval_query(
            "O que é RAG?",
            generate_english=fail,
        )

    assert result == "O que é RAG?"
    assert "English retrieval rewrite failed" in caplog.text


def test_to_english_retrieval_query_uses_gemini(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            self.models = self

        def generate_content(self, **kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(text="what is RAG retrieval augmented generation")

    monkeypatch.setattr(english_retrieval.genai, "Client", FakeClient)

    result = to_english_retrieval_query(
        "O que é RAG?",
        settings=Settings(gemini_api_key="test-key"),
    )

    assert result == "what is RAG retrieval augmented generation"


def test_to_english_retrieval_query_falls_back_when_gemini_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quota_error = genai_errors.APIError(
        429,
        {
            "error": {
                "message": "You exceeded your current quota",
                "status": "RESOURCE_EXHAUSTED",
            }
        },
    )

    class FailingClient:
        def __init__(self, **kwargs: Any) -> None:
            self.models = self

        def generate_content(self, **kwargs: Any) -> None:
            raise quota_error

    monkeypatch.setattr(english_retrieval.genai, "Client", FailingClient)

    result = to_english_retrieval_query(
        "O que é RAG?",
        settings=Settings(gemini_api_key="test-key"),
    )

    assert result == "O que é RAG?"
