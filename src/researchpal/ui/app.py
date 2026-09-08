"""Gradio ChatInterface that forwards each message to POST /ask."""

from __future__ import annotations

import os
from collections.abc import Sequence

import gradio as gr

from researchpal.ui.client import ResearchPalUIError, ask_http

DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_ASK_TIMEOUT = 180.0


def _api_url() -> str:
    return os.environ.get("RESEARCHPAL_API_URL", DEFAULT_API_URL).strip() or DEFAULT_API_URL


def _ask_timeout() -> float:
    raw = os.environ.get("RESEARCHPAL_ASK_TIMEOUT", str(DEFAULT_ASK_TIMEOUT))
    try:
        timeout = float(raw)
    except ValueError as error:
        raise ResearchPalUIError(
            "RESEARCHPAL_ASK_TIMEOUT must be a number of seconds"
        ) from error
    if timeout <= 0:
        raise ResearchPalUIError("RESEARCHPAL_ASK_TIMEOUT must be greater than zero")
    return timeout


def respond(message: str, history: Sequence[object] | None = None) -> str:
    """Send only the latest user message to `/ask`. History is display-only."""
    _ = history
    try:
        return ask_http(message, base_url=_api_url(), timeout=_ask_timeout())
    except ResearchPalUIError as error:
        raise gr.Error(str(error)) from error


def build_interface() -> gr.ChatInterface:
    """Build the chat page without launching a server."""
    return gr.ChatInterface(
        respond,
        title="ResearchPal",
        description=(
            "Ask questions about the ingested arXiv papers. "
            "Each message is an independent question; the API does not keep chat memory."
        ),
    )


def launch() -> None:
    """Start the Gradio server."""
    build_interface().launch()
