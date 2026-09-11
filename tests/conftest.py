import pytest


@pytest.fixture(autouse=True)
def passthrough_english_retrieval_query(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tool/agent tests offline: rewrite is covered in test_english_retrieval."""
    monkeypatch.setattr(
        "researchpal.tools.document_tools.to_english_retrieval_query",
        lambda query, settings=None: query,
    )
