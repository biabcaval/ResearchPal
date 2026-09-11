"""Semantic search tool over indexed English paper chunks."""

import logging

import chromadb

from researchpal.config import Settings, get_settings
from researchpal.models import RetrievedDocument, SearchToolParams, ToolResult
from researchpal.tools.base import ResearchTool
from researchpal.tools.english_retrieval import to_english_retrieval_query
from researchpal.tools.vector_store import VectorStore

logger = logging.getLogger(__name__)


class SearchDocumentsTool(ResearchTool[SearchToolParams, list[RetrievedDocument]]):
    """Search ChromaDB for the chunks most relevant to one query."""

    name = "search_documents"
    description = (
        "Busca semanticamente os chunks mais relevantes dos artigos indexados. "
        "O índice é em inglês: passe a query em inglês. Se a pergunta do usuário "
        "estiver em português, reescreva-a como uma query de busca em inglês. "
        "A resposta ao usuário continua em português. Use quando precisar "
        "encontrar evidência textual para responder à pergunta."
    )
    params_model = SearchToolParams

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        store: VectorStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or VectorStore(
            self.settings.chroma_path,
            self.settings.collection_name,
        )

    def run(
        self,
        params: SearchToolParams,
    ) -> ToolResult[list[RetrievedDocument]]:
        """Rewrite the query to English, then search the vector store."""
        try:
            retrieval_query = to_english_retrieval_query(
                params.query,
                settings=self.settings,
            )
            documents = self.store.search(retrieval_query, params.limit)
        except (chromadb.errors.ChromaError, RuntimeError, ValueError) as error:
            logger.warning("Document search failed: %s", error)
            return ToolResult(success=False, error=f"Document search failed: {error}")
        return ToolResult(success=True, data=documents)
