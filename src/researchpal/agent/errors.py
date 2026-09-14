"""Agent-level exceptions."""

from google.genai import errors as genai_errors
from langchain_core.exceptions import ModelError
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

GEMINI_UPSTREAM_ERRORS = (
    genai_errors.APIError,
    genai_errors.ClientError,
    ModelError,
    ChatGoogleGenerativeAIError,
)


class ModelUnavailableError(RuntimeError):
    """Raised when Gemini rejects or fails a request (quota, overload, transport).

    Kept distinct from a genuine lack of evidence so the HTTP layer can report an
    upstream outage instead of claiming the corpus has no answer.
    """
