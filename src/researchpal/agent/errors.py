"""Agent-level exceptions."""


class ModelUnavailableError(RuntimeError):
    """Raised when Gemini rejects or fails a request (quota, overload, transport).

    Kept distinct from a genuine lack of evidence so the HTTP layer can report an
    upstream outage instead of claiming the corpus has no answer.
    """
