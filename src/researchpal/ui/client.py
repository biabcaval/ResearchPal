"""HTTP client used by the Gradio UI to call POST /ask."""

from typing import Any

import requests


class ResearchPalUIError(Exception):
    """Raised when the UI cannot obtain an answer from the API."""


def ask_http(question: str, *, base_url: str, timeout: float) -> str:
    """POST a question to `/ask` and return the answer string.

    Args:
        question: User question. Whitespace-only values are rejected.
        base_url: FastAPI origin, for example `http://127.0.0.1:8000`.
        timeout: Requests timeout in seconds.

    Returns:
        The `answer` field from a successful JSON response.

    Raises:
        ResearchPalUIError: Empty question, network failure, or non-success API result.
    """
    stripped = question.strip()
    if not stripped:
        raise ResearchPalUIError("Question must not be empty")

    url = f"{base_url.rstrip('/')}/ask"
    try:
        response = requests.post(
            url,
            json={"question": stripped},
            timeout=timeout,
        )
    except requests.Timeout as error:
        raise ResearchPalUIError("The ResearchPal API timed out") from error
    except requests.ConnectionError as error:
        raise ResearchPalUIError(
            "Could not reach the ResearchPal API. Is uvicorn running?"
        ) from error

    if response.status_code != 200:
        raise ResearchPalUIError(
            f"API error {response.status_code}: {_response_detail(response)}"
        )

    payload: Any
    try:
        payload = response.json()
    except ValueError as error:
        raise ResearchPalUIError(
            "API error: response did not include an answer"
        ) from error

    answer = payload.get("answer") if isinstance(payload, dict) else None
    if not isinstance(answer, str):
        raise ResearchPalUIError("API error: response did not include an answer")
    return answer


def _response_detail(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text
    if isinstance(payload, dict) and "detail" in payload:
        return str(payload["detail"])
    return response.text
