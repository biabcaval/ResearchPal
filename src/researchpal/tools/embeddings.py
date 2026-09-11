"""English MiniLM encoder shared by ingestion and query."""

from chromadb.api.types import EmbeddingFunction
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

ENGLISH_MINILM_MODEL = "all-MiniLM-L6-v2"


def paper_embedding_function() -> EmbeddingFunction:
    """Return Chroma's English ONNX MiniLM encoder (`all-MiniLM-L6-v2`).

    Papers are indexed in English. Search queries are rewritten to English
    before comparison, so both sides of cosine similarity stay in English.
    """
    return DefaultEmbeddingFunction()
