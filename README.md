# ResearchPal

Question-and-answer agent for systematic literature review over a fixed set of arXiv papers.

## Architecture overview

ResearchPal is a local RAG service: an HTTP API receives a question, a Gemini agent decides which tools to call, and those tools retrieve evidence from a persistent ChromaDB index and from locally stored PDFs. The agent then synthesizes an answer in Portuguese from that evidence only.

```text
                    ┌─────────────────────────┐
                    │   HTTP client / Swagger │
                    └────────────┬────────────┘
                                 │ POST /ask  {"question": "..."}
                                 v
                    ┌─────────────────────────┐
                    │ FastAPI  (app.main:app) │
                    │ validation + HTTP deps  │
                    └────────────┬────────────┘
                                 │
                                 v
                    ┌─────────────────────────┐
                    │   GeminiResearchAgent   │
                    │  google-genai function  │
                    │  calling (max 3 rounds) │
                    └──────┬──────────┬───────┘
           search_documents│          │extract_section
                           v          v
                 ┌──────────────┐  ┌──────────────────┐
                 │   ChromaDB   │  │  Local PDFs      │
                 │ data/chroma  │  │  data/pdfs       │
                 └──────────────┘  └──────────────────┘
                           ^
                           │ ingest.py
                 ┌─────────┴─────────┐
                 │ Ingestion pipeline│
                 │ arXiv → PDF text  │
                 │ → overlapping     │
                 │   token chunks    │
                 └───────────────────┘
```

Package layout:

- `researchpal/agent`: Gemini orchestration and function-calling loop.
- `researchpal/tools`: ChromaDB search and PDF section extraction.
- `researchpal/api`: FastAPI application and HTTP dependencies.
- `researchpal/models`: internal request, tool, and RAG models.
- `researchpal/config`: environment configuration.
- `researchpal/pipeline`: PDF download, extraction, and ingestion.

`POST /ask` accepts `{"question": "..."}` and returns `{"question": "...", "answer": "..."}`. Interactive docs are at `http://127.0.0.1:8000/docs`. The Gradio UI (`ui.py`) is a thin HTTP client of that endpoint.

## Tools vs agent

**Tools** are deterministic, stateless functions. They do not choose a workflow and they do not generate prose. Each tool takes validated parameters, talks to one data source, and returns a structured `ToolResult`:

- `search_documents`: semantic query against ChromaDB chunks.
- `extract_section`: reads a local PDF and returns only `abstract`, `introduction`, or `conclusion`.

**The agent** (`GeminiResearchAgent`) is the decision layer. It receives the user question, exposes the tools to Gemini via function declarations, executes the calls Gemini requests, and asks the model to synthesize a Portuguese answer from the collected evidence. It never invents citations or paper content; if tools return nothing, it says there is not enough evidence.

This split exists so retrieval stays testable and replaceable (ChromaDB and PDFs can be mocked) while the LLM only reasons over tool output. The HTTP layer does not open ChromaDB or Gemini itself; FastAPI dependencies can be overridden in tests.

## Setup

Requirements: Python 3.12+, [uv](https://docs.astral.sh/uv/), and a Google AI Studio API key.

This project is managed with `pyproject.toml`. Install dependencies with `uv sync`; do not expect a `requirements.txt` workflow.

```bash
uv sync --dev
cp .env.example .env
# Edit .env and set GEMINI_API_KEY
uv run python ingest.py
uv run uvicorn app.main:app --reload
```

The Gradio UI is a separate process that calls `POST /ask`. Keep uvicorn running, then in another terminal:

```bash
uv run python ui.py
```

Open the URL Gradio prints (usually `http://127.0.0.1:7860`). Each chat message is an independent question. Override the API origin with `RESEARCHPAL_API_URL` (default `http://127.0.0.1:8000`).

On Windows, copy the env file with `Copy-Item .env.example .env`.

`uv run` is the supported way to execute project scripts and the API inside the synced environment. Ingestion downloads exactly three arXiv PDFs (`1706.03762`, `1810.04805`, `2005.11401`) into `data/pdfs` and upserts chunks into `data/chroma`. Re-run ingestion after changing the embedding model so Chroma does not keep vectors from a previous encoder.

Run tests (no network, no real ChromaDB, no Gemini):

```bash
uv run pytest
```

## Technical decisions

**Vector store — ChromaDB (persistent client).** The corpus is three papers on disk, so an embedded store under `data/chroma` avoids operating a separate database. Persistence means ingestion is a one-shot CLI (`ingest.py`) and the API only reads the same collection. Chunk IDs are deterministic (`{paper_id}-page-{n}-chunk-{i}`), so `upsert` is idempotent and re-running ingestion overwrites the same documents.

**Chunking — fixed-size token windows with overlap.** Each PDF page is split into 256-token chunks with 32-token overlap (`RESEARCHPAL_CHUNK_SIZE` / `RESEARCHPAL_CHUNK_OVERLAP`) using `tiktoken` `cl100k_base`. Token windows match how embedding models budget context (MiniLM is ~256 tokens), overlap preserves sentences that straddle a cut, and page metadata stays attached to every chunk for citations. Empty PDFs, pages with no extractable text, and download failures abort ingestion with an explicit error.

**Embedding model — English MiniLM, Portuguese answers.** The store uses Chroma's default ONNX MiniLM encoder (`all-MiniLM-L6-v2`). Papers stay in English. `search_documents` rewrites the question into an English search query so comparison is English-to-English, then Gemini answers in Portuguese (the evaluation language). English questions are also rewritten into a retrieval query; the user-facing answer stays in Portuguese. Re-run `ingest.py` after this encoder change if the collection was built with multilingual MiniLM.

**LLM — official `google-genai` SDK, default `gemini-3.5-flash`.** Function calling is implemented with `google-genai` (`>=1.0.0`), not a third-party agent framework, so tool schemas stay aligned with Pydantic models. Flash is the default for latency and cost on a three-paper corpus; the name is configurable via `GEMINI_MODEL`. Temperature is `0.0` and the tool loop is capped at three rounds.

## Known limitations

- The corpus is closed: only the three selected arXiv IDs are ingested and searchable.
- Answers depend on PyPDF2 text extraction; layout, figures, and equations are not recovered as structured content.
- Section extraction is heading-regex based (`abstract` / `introduction` / `conclusion`) and can miss papers with unusual headings.
- The agent has no memory across requests and stops calling tools after three rounds.
- Retrieval quality is bounded by MiniLM embeddings, chunk size, and `RESEARCHPAL_RETRIEVAL_LIMIT` (default 5). There is no hybrid BM25 + vector search. If the English query rewrite fails, search falls back to the original question and recall can drop.
- The public HTTP contract returns only `question` and `answer`; sources and tool errors exist internally but are not exposed to clients. Gemini failures (429 quota, 503 overload) are the exception: they return HTTP `503` with the upstream reason instead of an answer, so a quota problem is never disguised as missing evidence.
- `GEMINI_API_KEY` is required at request time (`503` if missing). Importing the app does not open the network or the database.

---

🇧🇷 **pt-br**

---

# ResearchPal

Agente de perguntas e respostas para revisão sistemática da literatura sobre um conjunto fixo de artigos do arXiv.

## Visão geral da arquitetura

O ResearchPal é um serviço RAG local: a API HTTP recebe a pergunta, um agente Gemini decide quais tools chamar, e essas tools recuperam evidência de um índice persistente no ChromaDB e de PDFs armazenados localmente. O agente sintetiza a resposta em português usando apenas essa evidência.

```text
                    ┌─────────────────────────┐
                    │   Cliente HTTP / Swagger│
                    └────────────┬────────────┘
                                 │ POST /ask  {"question": "..."}
                                 v
                    ┌─────────────────────────┐
                    │ FastAPI  (app.main:app) │
                    │ validação + deps HTTP   │
                    └────────────┬────────────┘
                                 │
                                 v
                    ┌─────────────────────────┐
                    │   GeminiResearchAgent   │
                    │  function calling       │
                    │  google-genai (máx. 3)  │
                    └──────┬──────────┬───────┘
           search_documents│          │extract_section
                           v          v
                 ┌──────────────┐  ┌──────────────────┐
                 │   ChromaDB   │  │  PDFs locais     │
                 │ data/chroma  │  │  data/pdfs       │
                 └──────────────┘  └──────────────────┘
                           ^
                           │ ingest.py
                 ┌─────────┴─────────┐
                 │ Pipeline de ingestão│
                 │ arXiv → texto PDF │
                 │ → chunks de       │
                 │   tokens com      │
                 │   sobreposição    │
                 └───────────────────┘
```

Organização do pacote:

- `researchpal/agent`: orquestração Gemini e loop de function calling.
- `researchpal/tools`: busca no ChromaDB e extração de seções do PDF.
- `researchpal/api`: aplicação FastAPI e dependências HTTP.
- `researchpal/models`: modelos internos de request, tools e RAG.
- `researchpal/config`: configuração via ambiente.
- `researchpal/pipeline`: download, extração e ingestão de PDFs.

`POST /ask` aceita `{"question": "..."}` e devolve `{"question": "...", "answer": "..."}`. A documentação interativa fica em `http://127.0.0.1:8000/docs`. A UI Gradio (`ui.py`) é um cliente HTTP desse endpoint.

## Distinção entre tools e agente

**Tools** são funções determinísticas e sem estado. Elas não escolhem o fluxo e não geram prosa. Cada uma recebe parâmetros validados, acessa uma única fonte de dados e devolve um `ToolResult` estruturado:

- `search_documents`: consulta semântica nos chunks do ChromaDB.
- `extract_section`: lê um PDF local e devolve somente `abstract`, `introduction` ou `conclusion`.

**O agente** (`GeminiResearchAgent`) é a camada de decisão. Ele recebe a pergunta, expõe as tools ao Gemini via function declarations, executa as chamadas pedidas pelo modelo e pede uma síntese em português a partir da evidência coletada. Não inventa citações nem conteúdo dos artigos; se as tools não devolverem evidência, declara que não há base suficiente.

Essa separação existe para que a recuperação continue testável e substituível (ChromaDB e PDFs podem ser mockados), enquanto o LLM raciocina só sobre a saída das tools. A camada HTTP não abre o ChromaDB nem o Gemini; as dependências do FastAPI podem ser substituídas nos testes.

## Instruções de setup

Requisitos: Python 3.12+, [uv](https://docs.astral.sh/uv/) e uma chave da Google AI Studio.

O projeto é gerenciado com `pyproject.toml`. Instale as dependências com `uv sync`; não há fluxo baseado em `requirements.txt`.

```bash
uv sync --dev
cp .env.example .env
# Edite o .env e defina GEMINI_API_KEY
uv run python ingest.py
uv run uvicorn app.main:app --reload
```

A UI Gradio é um processo separado que chama `POST /ask`. Com o uvicorn no ar, em outro terminal:

```bash
uv run python ui.py
```

Abra a URL que o Gradio imprimir (em geral `http://127.0.0.1:7860`). Cada mensagem do chat é uma pergunta independente. Sobrescreva a origem da API com `RESEARCHPAL_API_URL` (padrão `http://127.0.0.1:8000`).

No Windows, copie o arquivo de ambiente com `Copy-Item .env.example .env`.

`uv run` é a forma suportada de executar scripts e a API no ambiente sincronizado. A ingestão baixa exatamente três PDFs do arXiv (`1706.03762`, `1810.04805`, `2005.11401`) em `data/pdfs` e faz upsert dos chunks em `data/chroma`. Reexecute a ingestão depois de trocar o modelo de embedding para o Chroma não manter vetores de um encoder anterior.

Testes (sem rede, sem ChromaDB real, sem Gemini):

```bash
uv run pytest
```

## Decisões técnicas

**Vector store — ChromaDB (cliente persistente).** O corpus são três artigos em disco, então um store embutido em `data/chroma` evita operar um banco separado. A persistência faz da ingestão um CLI único (`ingest.py`) e a API só lê a mesma collection. Os IDs dos chunks são determinísticos (`{paper_id}-page-{n}-chunk-{i}`), então o `upsert` é idempotente e reexecutar a ingestão sobrescreve os mesmos documentos.

**Chunking — janelas de tokens com sobreposição.** Cada página do PDF é dividida em chunks de 256 tokens com sobreposição de 32 (`RESEARCHPAL_CHUNK_SIZE` / `RESEARCHPAL_CHUNK_OVERLAP`) usando `tiktoken` `cl100k_base`. Janelas em tokens acompanham o orçamento dos embeddings (MiniLM ~256 tokens), a sobreposição preserva frases cortadas no limite, e os metadados de página permanecem em cada chunk para citação. PDFs vazios, páginas sem texto extraível e falhas de download interrompem a ingestão com erro explícito.

**Modelo de embedding — MiniLM inglês, respostas em português.** O store usa o encoder ONNX MiniLM padrão do Chroma (`all-MiniLM-L6-v2`). Os artigos permanecem em inglês. `search_documents` reescreve a pergunta como uma query de busca em inglês para a comparação ser inglês-com-inglês; o Gemini responde em português (a língua da avaliação). Perguntas já em inglês também são reescritas para retrieval; a resposta ao usuário continua em português. Rode `ingest.py` de novo se a collection foi criada com MiniLM multilingual.

**LLM — SDK oficial `google-genai`, padrão `gemini-3.5-flash`.** O function calling usa `google-genai` (`>=1.0.0`), sem um framework de agentes de terceiros, para manter os schemas das tools alinhados aos modelos Pydantic. Flash é o padrão por latência e custo em um corpus de três artigos; o nome é configurável em `GEMINI_MODEL`. A temperatura é `0.0` e o loop de tools tem no máximo três rodadas.

## Limitações conhecidas

- O corpus é fechado: só os três IDs obrigatórios do arXiv são ingeridos e pesquisáveis.
- As respostas dependem da extração de texto do PyPDF2; layout, figuras e equações não são recuperados como conteúdo estruturado.
- A extração de seções usa regex de headings (`abstract` / `introduction` / `conclusion`) e pode falhar em artigos com títulos atípicos.
- O agente não tem memória entre requisições e para de chamar tools após três rodadas.
- A qualidade da recuperação é limitada pelos embeddings MiniLM, pelo tamanho do chunk e por `RESEARCHPAL_RETRIEVAL_LIMIT` (padrão 5). Não há busca híbrida BM25 + vetorial. Se a reescrita da query para inglês falhar, a busca cai na pergunta original e o recall pode cair.
- O contrato HTTP público devolve só `question` e `answer`; fontes e erros de tools existem internamente, mas não são expostos ao cliente. As falhas do Gemini (429 de cota, 503 de sobrecarga) são a exceção: devolvem HTTP `503` com o motivo original, para que um problema de cota nunca seja confundido com falta de evidência.
- `GEMINI_API_KEY` é obrigatória na hora da requisição (`503` se estiver ausente). Importar a aplicação não abre a rede nem o banco.
