import pytest

from researchpal.tools import embeddings


def test_paper_embedding_function_uses_english_onnx_minilm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    monkeypatch.setattr(embeddings, "DefaultEmbeddingFunction", lambda: sentinel)

    result = embeddings.paper_embedding_function()

    assert result is sentinel
    assert embeddings.ENGLISH_MINILM_MODEL == "all-MiniLM-L6-v2"
