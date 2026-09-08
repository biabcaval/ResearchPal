from unittest.mock import patch

import gradio as gr
import pytest

from researchpal.ui.app import respond
from researchpal.ui.client import ResearchPalUIError


def test_respond_sends_only_the_latest_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RESEARCHPAL_API_URL", raising=False)
    monkeypatch.delenv("RESEARCHPAL_ASK_TIMEOUT", raising=False)
    with patch(
        "researchpal.ui.app.ask_http",
        return_value="resposta",
    ) as ask:
        answer = respond(
            "pergunta nova",
            history=[{"role": "user", "content": "pergunta antiga"}],
        )

    assert answer == "resposta"
    ask.assert_called_once_with(
        "pergunta nova",
        base_url="http://127.0.0.1:8000",
        timeout=180.0,
    )


def test_respond_raises_gradio_error_on_client_failure() -> None:
    with patch(
        "researchpal.ui.app.ask_http",
        side_effect=ResearchPalUIError("Could not reach the ResearchPal API. Is uvicorn running?"),
    ):
        with pytest.raises(gr.Error, match="uvicorn"):
            respond("pergunta")
