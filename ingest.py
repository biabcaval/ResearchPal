import logging

from researchpal.pipeline.ingestion import ingest_papers

logger = logging.getLogger(__name__)


def main() -> None:
    import argparse

    argparse.ArgumentParser(
        description="Ingest the three required arXiv papers into ChromaDB."
    ).parse_args()
    identifiers = ingest_papers()
    logger.info("Ingested %s paper(s): %s", len(identifiers), ", ".join(identifiers))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    main()
