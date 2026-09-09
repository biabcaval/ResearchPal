"""Gradio ChatInterface that forwards each message to POST /ask."""

from __future__ import annotations

from collections.abc import Sequence

import gradio as gr
from pydantic import ValidationError

from researchpal.config import get_settings
from researchpal.ui.client import ResearchPalUIError, ask_http


def respond(message: str, history: Sequence[object] | None = None) -> str:
    """Send only the latest user message to `/ask`. History is display-only."""
    _ = history
    try:
        settings = get_settings()
        return ask_http(
            message,
            base_url=settings.api_url,
            timeout=settings.ask_timeout,
        )
    except ValidationError as error:
        raise gr.Error(str(error)) from error
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
