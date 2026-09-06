from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REQUIRED_ARXIV_IDS = ("1706.03762", "1810.04805", "2005.11401")
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    chroma_path: Path = PROJECT_ROOT / "data" / "chroma"
    collection_name: str = "papers"
    pdf_directory: Path = PROJECT_ROOT / "data" / "pdfs"
    request_timeout: int = 30
    download_chunk_size: int = 1024 * 1024

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            chroma_path=Path(os.getenv("RESEARCHPAL_CHROMA_PATH", str(cls.chroma_path))),
            collection_name=os.getenv("RESEARCHPAL_COLLECTION", cls.collection_name),
            pdf_directory=Path(
                os.getenv("RESEARCHPAL_PDF_DIRECTORY", str(cls.pdf_directory))
            ),
            request_timeout=int(
                os.getenv("RESEARCHPAL_REQUEST_TIMEOUT", str(cls.request_timeout))
            ),
            download_chunk_size=int(
                os.getenv("RESEARCHPAL_DOWNLOAD_CHUNK_SIZE", str(cls.download_chunk_size))
            ),
        )


def get_settings() -> Settings:
    return Settings.from_environment()
