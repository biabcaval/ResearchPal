"""HTTP client used by the Gradio UI to call POST /ask."""

import requests
from pydantic import ValidationError

from researchpal.api.schemas import AskHttpResponse
from researchpal.models import AskRequest


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
            json=AskRequest(question=stripped).model_dump(),
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

    try:
        payload = response.json()
        parsed = AskHttpResponse.model_validate(payload)
    except (ValueError, ValidationError) as error:
        raise ResearchPalUIError(
            "API error: response did not include an answer"
        ) from error
    return parsed.answer


def _response_detail(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text
    if isinstance(payload, dict) and "detail" in payload:
        return str(payload["detail"])
    return response.text
