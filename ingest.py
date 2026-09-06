from researchpal.pipeline.ingestion import ingest_papers


def main() -> None:
    import argparse

    argparse.ArgumentParser(
        description="Ingest the three required arXiv papers into ChromaDB."
    ).parse_args()
    identifiers = ingest_papers()
    print(f"Ingested {len(identifiers)} paper(s): {', '.join(identifiers)}")


if __name__ == "__main__":
    main()
