from unittest.mock import Mock, patch

import gradio as gr
import pytest

from researchpal.api.schemas import AskHttpResponse
from researchpal.config import Settings
from researchpal.models import Citation
from researchpal.ui.app import respond
from researchpal.ui.client import ResearchPalUIError


def test_respond_sends_only_the_latest_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "researchpal.ui.app.get_settings",
        lambda: Settings(
            gemini_api_key="test",
            api_url="http://127.0.0.1:8000",
            ask_timeout=180.0,
        ),
    )
    with patch(
        "researchpal.ui.app.ask_http",
        return_value=AskHttpResponse(
            question="pergunta nova",
            answer="resposta [1]",
            citations=[
                Citation(
                    number=1,
                    identifier="1706.03762-page-1-chunk-0",
                    paper_id="1706.03762",
                    page=1,
                    snippet="Attention improves sequence modeling.",
                )
            ],
        ),
    ) as ask:
        answer = respond(
            "pergunta nova",
            history=[{"role": "user", "content": "pergunta antiga"}],
        )

    assert 'class="rp-cite"' in answer
    assert "1706.03762" in answer
    ask.assert_called_once_with(
        "pergunta nova",
        base_url="http://127.0.0.1:8000",
        timeout=180.0,
    )


def test_respond_uses_api_url_and_timeout_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "researchpal.ui.app.get_settings",
        lambda: Settings(
            gemini_api_key="test",
            api_url="http://api.local:9000",
            ask_timeout=30.0,
        ),
    )
    with patch(
        "researchpal.ui.app.ask_http",
        return_value=AskHttpResponse(
            question="pergunta",
            answer="ok",
            citations=[],
        ),
    ) as ask:
        respond("pergunta")

    ask.assert_called_once_with(
        "pergunta",
        base_url="http://api.local:9000",
        timeout=30.0,
    )


def test_respond_raises_gradio_error_on_client_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "researchpal.ui.app.get_settings",
        lambda: Settings(gemini_api_key="test"),
    )
    with (
        patch(
            "researchpal.ui.app.ask_http",
            side_effect=ResearchPalUIError(
                "Could not reach the ResearchPal API. Is uvicorn running?"
            ),
        ),
        pytest.raises(gr.Error, match="uvicorn"),
    ):
        respond("pergunta")


def test_launch_passes_citation_css() -> None:
    interface = Mock()
    with (
        patch("researchpal.ui.app.build_interface", return_value=interface),
        patch("researchpal.ui.app.CITATION_CSS", "/* cite */"),
    ):
        from researchpal.ui.app import launch

        launch()

    interface.launch.assert_called_once_with(css="/* cite */")
