from unittest.mock import Mock, patch

import pytest
import requests

from researchpal.ui.client import ResearchPalUIError, ask_http


def test_ask_http_returns_answer_from_successful_response() -> None:
    response = Mock()
    response.status_code = 200
    response.json.return_value = {
        "question": "O que é atenção?",
        "answer": "Atenção é um mecanismo...",
    }

    with patch("researchpal.ui.client.requests.post", return_value=response) as post:
        answer = ask_http(
            "O que é atenção?",
            base_url="http://127.0.0.1:8000",
            timeout=180,
        )

    assert answer == "Atenção é um mecanismo..."
    post.assert_called_once_with(
        "http://127.0.0.1:8000/ask",
        json={"question": "O que é atenção?"},
        timeout=180,
    )


def test_ask_http_strips_trailing_slash_from_base_url() -> None:
    response = Mock()
    response.status_code = 200
    response.json.return_value = {"question": "q", "answer": "a"}

    with patch("researchpal.ui.client.requests.post", return_value=response) as post:
        ask_http("q", base_url="http://127.0.0.1:8000/", timeout=10)

    post.assert_called_once_with(
        "http://127.0.0.1:8000/ask",
        json={"question": "q"},
        timeout=10,
    )


def test_ask_http_does_not_post_when_question_is_empty() -> None:
    with (
        patch("researchpal.ui.client.requests.post") as post,
        pytest.raises(ResearchPalUIError, match="empty"),
    ):
        ask_http("   ", base_url="http://127.0.0.1:8000", timeout=10)

    post.assert_not_called()


def test_ask_http_maps_connection_error() -> None:
    with patch(
        "researchpal.ui.client.requests.post",
        side_effect=requests.ConnectionError("refused"),
    ), pytest.raises(ResearchPalUIError, match="uvicorn"):
        ask_http("pergunta", base_url="http://127.0.0.1:8000", timeout=10)


def test_ask_http_maps_timeout() -> None:
    with patch(
        "researchpal.ui.client.requests.post",
        side_effect=requests.Timeout("slow"),
    ), pytest.raises(ResearchPalUIError, match="timed out"):
        ask_http("pergunta", base_url="http://127.0.0.1:8000", timeout=10)


def test_ask_http_rejects_response_missing_question() -> None:
    response = Mock()
    response.status_code = 200
    response.json.return_value = {"answer": "partial payload"}

    with (
        patch("researchpal.ui.client.requests.post", return_value=response),
        pytest.raises(ResearchPalUIError, match="did not include an answer"),
    ):
        ask_http("pergunta", base_url="http://127.0.0.1:8000", timeout=10)


def test_ask_http_maps_503_detail() -> None:
    response = Mock()
    response.status_code = 503
    response.json.return_value = {"detail": "GEMINI_API_KEY is not configured"}
    response.text = "ignored"

    with (
        patch("researchpal.ui.client.requests.post", return_value=response),
        pytest.raises(ResearchPalUIError, match="API error 503: GEMINI_API_KEY"),
    ):
        ask_http("pergunta", base_url="http://127.0.0.1:8000", timeout=10)
