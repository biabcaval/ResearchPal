"""Rewrite search queries into English before MiniLM comparison."""

from __future__ import annotations

import logging
from collections.abc import Callable

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from researchpal.config import Settings, get_settings

logger = logging.getLogger(__name__)

REWRITE_INSTRUCTION = """
Rewrite the text as an English search query for retrieving English academic papers.
Use paper vocabulary such as methods, findings, and model names.
Keep technical terms, paper titles, and identifiers unchanged.
If the text is already English, return a concise English search query.
Return only the query, with no quotes or explanation.
"""


def to_english_retrieval_query(
    query: str,
    *,
    settings: Settings | None = None,
    generate_english: Callable[[str], str] | None = None,
) -> str:
    """Rewrite a user or tool query into English for retrieval.

    Portuguese and English questions are both rewritten so Chroma compares
    English text with the original English paper chunks. Rewrite failures
    fall back to the original query.

    Args:
        query: Search text from the user or from `search_documents`.
        settings: Runtime settings; loaded from the environment when omitted.
        generate_english: Optional rewriter used in tests instead of Gemini.

    Returns:
        English retrieval text, or the original query if rewriting fails.
    """
    stripped = query.strip()
    if not stripped:
        return stripped

    try:
        if generate_english is not None:
            rewritten = generate_english(stripped)
        else:
            rewritten = _rewrite_with_gemini(stripped, settings or get_settings())
    except (
        OSError,
        RuntimeError,
        ValueError,
        genai_errors.APIError,
        genai_errors.ClientError,
    ) as error:
        logger.warning("English retrieval rewrite failed; using original query: %s", error)
        return stripped

    cleaned = _clean_rewritten_query(rewritten)
    if not cleaned:
        logger.warning("English retrieval rewrite returned empty text; using original query")
        return stripped
    return cleaned


def _rewrite_with_gemini(query: str, settings: Settings) -> str:
    """Ask Gemini to rewrite one retrieval query into English."""
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    client = genai.Client(api_key=settings.gemini_api_key)
    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=query,
        config=types.GenerateContentConfig(
            system_instruction=REWRITE_INSTRUCTION,
            temperature=0.0,
        ),
    )
    return _clean_rewritten_query(response.text or "")


def _clean_rewritten_query(text: str) -> str:
    """Strip wrapping quotes the model sometimes adds around the query."""
    cleaned = text.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
        return cleaned[1:-1].strip()
    return cleaned
