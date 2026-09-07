from researchpal.config import Settings, get_settings
from researchpal.models import RetrievedDocument, SearchToolParams
from researchpal.tools import VectorStore


class ResearchAgent:
    def __init__(self, settings: Settings | None = None) -> None:
        active_settings = settings or get_settings()
        self.store = VectorStore(active_settings.chroma_path, active_settings.collection_name)

    def search(self, params: SearchToolParams) -> list[RetrievedDocument]:
        return self.store.search(
            question=params.query,
            limit=params.limit,
        )
