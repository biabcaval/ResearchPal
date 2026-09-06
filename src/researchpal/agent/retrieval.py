from researchpal.config import Settings, get_settings
from researchpal.models import QueryResult
from researchpal.tools import VectorStore


class ResearchAgent:
    def __init__(self, settings: Settings | None = None) -> None:
        active_settings = settings or get_settings()
        self.store = VectorStore(active_settings.chroma_path, active_settings.collection_name)

    def search(self, question: str, limit: int = 5) -> QueryResult:
        result = self.store.query(question, limit)
        return QueryResult(
            ids=result.get("ids", []),
            distances=result.get("distances", []),
            raw=result,
        )
