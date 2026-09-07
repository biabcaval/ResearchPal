# ResearchPal

Question-and-answer agent for systematic literature review.

## Structure

- `researchpal/agent`: search retrieval and orchestration logic.
- `researchpal/tools`: access to ChromaDB and PDF extraction.
- `researchpal/api`: FastAPI application and HTTP dependencies.
- `researchpal/models`: internal agent and tool models.
- `researchpal/config`: environment configuration.
- `researchpal/pipeline`: PDF download, extraction, and ingestion.

## Architecture

```text
HTTP client
    |
    v
FastAPI (/ask) -- HTTP validation and serialization
    |
    v
GeminiResearchAgent -- decides when to call each tool and synthesizes the answer
    |                         |
    v                         v
search_documents        extract_section
    |                         |
    v                         v
ChromaDB                 Local PDFs
```

A **tool** executes a deterministic operation and returns structured evidence:
`search_documents` queries ChromaDB, and `extract_section` reads an allowed section
from a PDF. The **agent** chooses which tools to call via function calling, gathers
evidence, and asks Gemini for a synthesis in Portuguese. The HTTP layer does not
know about ChromaDB and does not create external clients directly; its dependencies
can be overridden in tests via `app.dependency_overrides`.

## Setup from scratch

Requirements: Python 3.12+, `uv`, and a Google AI Studio API key.

```bash
uv sync --dev
# Windows
Copy-Item .env.example .env
# Linux/macOS
# cp .env.example .env
# Edit the .env file and set GEMINI_API_KEY
uv run python ingest.py
uv run uvicorn app.main:app --reload
```

The `uv` equivalent to `pip install -r requirements.txt` is not a different concept;
it is the correct project workflow for this repository because the project is managed
with `pyproject.toml` instead of a `requirements.txt` file. The functional equivalent
is `uv sync --dev`, while running scripts and the API uses `uv run ...`.
The correct ASGI entrypoint is `app.main:app`, and the ingestion script is
`ingest.py` in the project root. The interactive API documentation is available at
`http://127.0.0.1:8000/docs`.

## API

`POST /ask` accepts only:

```json
{"question": "What do the papers say about attention?"}
```

and returns:

```json
{
  "question": "What do the papers say about attention?",
  "answer": "..."
}
```

Unknown fields or empty questions return `422`. Without `GEMINI_API_KEY`, the
agent's lazy initialization returns `503`; importing the app does not make network
calls or open the database. Swagger is available at `/docs`.

## Ingestion and execution

The pipeline processes exactly these IDs: `1706.03762`, `1810.04805`, and `2005.11401`.
PDFs are stored in `data/pdfs`, and ChromaDB is stored in `data/chroma` by default.
The ingestion step splits each page into chunks of 1000 characters with a 200-character
overlap (`RESEARCHPAL_CHUNK_SIZE` and `RESEARCHPAL_CHUNK_OVERLAP`).
Chunks receive deterministic IDs (`paper-page-chunk`), which makes the `upsert`
operation idempotent. Empty PDFs, pages with no text, and download failures stop the
ingestion process with an explicit error.

## Decisions and limitations

The agent uses the official `google-genai` library for function calling and configures
its model via `GEMINI_MODEL`. The agent does not retain memory between requests and
limits the function-calling loop to three rounds. The HTTP contract is smaller than
the internal agent model: sources and errors remain available internally without
coupling clients to the implementation format.

Retrieval depends on the locally ingested PDFs, and the answer quality depends on the
configured Gemini model and the PDF text extraction quality. Configure `GEMINI_API_KEY`
only in the `.env` file; no key is stored in the code or in the repository.

## Tests

The tests do not perform external calls: ChromaDB, PDFs, network access, and the
Gemini client are all mocked.

```bash
uv run pytest
```
