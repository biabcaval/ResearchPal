# ResearchPal

Question-and-answer agent for systematic literature review over a fixed set of arXiv papers.

## Architecture overview

ResearchPal is a local RAG service. Ingestion writes a persistent ChromaDB index from three arXiv PDFs. At query time, FastAPI validates `POST /ask`, a LangChain Gemini agent chooses tools, those tools retrieve evidence, a separate judge checks that the answer actually addresses the question, and citations are numbered before the HTTP response. The Gradio UI is an optional HTTP client of the same API.

```text
  ingest.py
    arXiv PDFs → PyPDF2 text → tiktoken token windows → ChromaDB (data/chroma)
    PDFs kept on disk (data/pdfs)

  Gradio (ui.py) ──┐
  Swagger / HTTP ──┤  POST /ask  {"question": "..."}
                   v
            FastAPI (app.main:app)
            validation + injected deps
                   v
         GeminiResearchAgent
            │  1. LangChain create_agent
            │     native Gemini function calling
            │     (1 tool-enabled model call)
            │  2. optional tool-free synthesis
            │  3. sanity judge (question + answer only)
            │  4. one rewrite if the judge fails
            │  5. [[chunk-id]] → numbered citations
            │
            ├── search_documents → ChromaDB chunks
            └── extract_section  → local PDF headings
                   v
         HTTP  {"question", "answer", "citations"}
            Gradio turns [n] into hover chips;
            if the answer has no [n], a Fontes row is appended.
```

Package layout:

- `researchpal/agent`: LangChain agent, sanity judge, citation numbering, Gemini error mapping. `AgentGraph`, `SynthesisModel`, and `AnswerSanityChecker` are protocols so tests inject doubles.
- `researchpal/tools`: ChromaDB search, English query rewrite, PDF section extraction. Tools expose LangChain `StructuredTool` adapters; they do not call the agent.
- `researchpal/api`: FastAPI application and HTTP dependencies.
- `researchpal/ui`: Gradio chat client and HTML citation chips.
- `researchpal/models`: internal request, tool, RAG, citation, and sanity models.
- `researchpal/config`: environment configuration (`pydantic-settings`).
- `researchpal/pipeline`: PDF download, extraction, and ingestion.

`POST /ask` accepts `{"question": "..."}` and returns `{"question": "...", "answer": "...", "citations": [...]}`. Numbered markers like `[1]` in `answer` map onto `citations`. Interactive docs are at `http://127.0.0.1:8000/docs`.

## Tools vs agent

**Tools** are deterministic, stateless classes. They do not choose a workflow, do not generate the user-facing answer, and do not decide whether that answer is complete. Each tool takes validated parameters, talks to one data source, and returns a structured `ToolResult`:

- `search_documents`: rewrites the question into an English retrieval query (`google-genai`), then runs a semantic search against ChromaDB chunks.
- `extract_section`: reads a local PDF and returns only `abstract`, `introduction`, or `conclusion`.

LangChain wraps those methods as `StructuredTool` so Gemini can call them with native function calling. That wrapper is I/O only: the same Python classes remain unit-testable without LangChain.

**The agent** (`GeminiResearchAgent`) is the decision and post-processing layer. It receives the user question, runs LangChain `create_agent` with native Gemini function calling (not ReAct), executes the tools Gemini requests, then:

1. May run one extra synthesis call with tools disabled (`ModelCallLimitMiddleware(run_limit=1)`).
2. Runs a structured Gemini sanity check on the candidate answer (on-topic and covers every asked-for part). Failures are rewritten once from the same evidence; a second failure becomes a fixed Portuguese fallback. Honest “not enough evidence” answers pass. The judge does not see retrieved chunks.
3. Maps `[[chunk-id]]` markers onto numbered `citations`. Unknown markers are stripped. If the model never cited, citations are still built from retrieved sources.

The agent never invents paper content. If tools return nothing, it says there is not enough evidence. FastAPI does not open ChromaDB or Gemini itself; dependencies (including the agent graph, synthesis model, and sanity checker) can be overridden in tests.

This split exists so retrieval stays replaceable and mockable, the LLM only reasons over tool output, and HTTP / UI stay thin clients of `/ask`.

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

**HTTP and UI — FastAPI + Gradio.** FastAPI owns validation, dependency injection, and the public JSON contract. Gradio (`ui.py`) is a ChatInterface that POSTs each message to `/ask` and renders citation chips; it does not hold retrieval state or call Gemini. Chat history in the UI is display-only.

**LLM — LangChain + Gemini 3.5 Flash.** The agent uses `langchain` `create_agent` with `langchain-google-genai` (`ChatGoogleGenerativeAI`) so Gemini native function calling drives `search_documents` and `extract_section`. LangChain is the LLM/tool-calling layer only — not document loaders, embeddings, or the Chroma retriever. ReAct is rejected because the challenge requires native function calling. Default model is `gemini-3.5-flash` (`GEMINI_MODEL`); Gemini 2.0 Flash is unavailable. Temperature is `0.0`. `ModelCallLimitMiddleware(run_limit=1)` caps tool-enabled calls, then one tool-free synthesis call when that limit fires.

**Citations — model markers, numbered API, hover UI.** The model must copy chunk identifiers as `[[id]]`. The agent maps known ids onto `[n]` plus `{number, paper_id, page, snippet}`. Gradio wraps those numbers as hover chips (paper id, page, snippet). If the answer has citations but no inline `[n]`, Gradio still shows a **Fontes** row so sources are visible.

**Answer sanity check — structured Gemini, no retrieval.** After a candidate answer exists, a separate structured call checks relevance and completeness. One rewrite is allowed from the same tool evidence. The judge is injected as `AnswerSanityChecker` so tests do not need Gemini.

## Known limitations

- The corpus is closed: only the three selected arXiv IDs are ingested and searchable.
- Answers depend on PyPDF2 text extraction; layout, figures, and equations are not recovered as structured content.
- Section extraction is heading-regex based (`abstract` / `introduction` / `conclusion`) and can miss papers with unusual headings.
- The agent has no memory across requests. It is allowed one tool-enabled model call, then at most one tool-free synthesis call.
- Retrieval quality is bounded by MiniLM embeddings, chunk size, and the search tool `limit` (default 5, set by the model). There is no hybrid BM25 + vector search. If the English query rewrite fails, search falls back to the original question and recall can drop.
- The public HTTP contract returns `question`, `answer`, and `citations`. Full retrieved chunks and tool errors stay internal. Gemini failures (429 quota, 503 overload) return HTTP `503` with the upstream reason instead of an answer, so a quota problem is never disguised as missing evidence.
- `/ask` may spend extra Gemini calls on the sanity judge and a single rewrite pass after a candidate answer exists. If the judge or rewrite hits quota or another upstream failure, the endpoint returns HTTP `503` even when a candidate answer was already available.
- Citation numbers only attach to `search_documents` chunk ids. `extract_section` output is not cited unless the same chunk id also appeared in search results. If the model omits `[[id]]` markers, the API still returns citations from retrieved sources, and Gradio appends Fontes.
- `GEMINI_API_KEY` is required at request time (`503` if missing). Importing the app does not open the network or the database.

---

🇧🇷 **pt-br**

---

# ResearchPal

Agente de perguntas e respostas para revisão sistemática da literatura sobre um conjunto fixo de artigos do arXiv.

## Visão geral da arquitetura

O ResearchPal é um serviço RAG local. A ingestão grava um índice persistente no ChromaDB a partir de três PDFs do arXiv. Na consulta, o FastAPI valida `POST /ask`, um agente Gemini (LangChain) escolhe as tools, as tools recuperam evidência, um juiz separado verifica se a resposta realmente atende à pergunta, e as citações são numeradas antes da resposta HTTP. A UI Gradio é um cliente HTTP opcional da mesma API.

```text
  ingest.py
    PDFs arXiv → texto PyPDF2 → janelas tiktoken → ChromaDB (data/chroma)
    PDFs ficam em disco (data/pdfs)

  Gradio (ui.py) ──┐
  Swagger / HTTP ──┤  POST /ask  {"question": "..."}
                   v
            FastAPI (app.main:app)
            validação + deps injetadas
                   v
         GeminiResearchAgent
            │  1. LangChain create_agent
            │     function calling nativo do Gemini
            │     (1 chamada de modelo com tools)
            │  2. síntese opcional sem tools
            │  3. juiz de sanidade (só pergunta + resposta)
            │  4. uma reescrita se o juiz falhar
            │  5. [[chunk-id]] → citações numeradas
            │
            ├── search_documents → chunks no ChromaDB
            └── extract_section  → headings do PDF local
                   v
         HTTP  {"question", "answer", "citations"}
            O Gradio transforma [n] em chips;
            se a resposta não tiver [n], acrescenta uma linha Fontes.
```

Organização do pacote:

- `researchpal/agent`: agente LangChain, juiz de sanidade, numeração de citações e mapeamento de erros do Gemini. `AgentGraph`, `SynthesisModel` e `AnswerSanityChecker` são protocols para testes injetarem doubles.
- `researchpal/tools`: busca no ChromaDB, reescrita da query para inglês e extração de seções do PDF. As tools expõem adapters `StructuredTool` do LangChain; elas não chamam o agente.
- `researchpal/api`: aplicação FastAPI e dependências HTTP.
- `researchpal/ui`: cliente Gradio e chips HTML de citação.
- `researchpal/models`: modelos internos de request, tools, RAG, citação e sanidade.
- `researchpal/config`: configuração via ambiente (`pydantic-settings`).
- `researchpal/pipeline`: download, extração e ingestão de PDFs.

`POST /ask` aceita `{"question": "..."}` e devolve `{"question": "...", "answer": "...", "citations": [...]}`. Marcadores `[1]` na resposta apontam para `citations`. A documentação interativa fica em `http://127.0.0.1:8000/docs`.

## Distinção entre tools e agente

**Tools** são classes determinísticas e sem estado. Elas não escolhem o fluxo, não geram a resposta ao usuário e não decidem se essa resposta está completa. Cada uma recebe parâmetros validados, acessa uma única fonte de dados e devolve um `ToolResult` estruturado:

- `search_documents`: reescreve a pergunta como query de retrieval em inglês (`google-genai`) e faz busca semântica nos chunks do ChromaDB.
- `extract_section`: lê um PDF local e devolve somente `abstract`, `introduction` ou `conclusion`.

O LangChain envolve esses métodos como `StructuredTool` para o Gemini chamá-los com function calling nativo. O wrapper é só I/O: as mesmas classes Python continuam testáveis sem LangChain.

**O agente** (`GeminiResearchAgent`) é a camada de decisão e pós-processamento. Ele recebe a pergunta, executa `create_agent` com function calling nativo do Gemini (não ReAct), roda as tools pedidas pelo modelo e depois:

1. Pode fazer uma síntese extra com tools desligadas (`ModelCallLimitMiddleware(run_limit=1)`).
2. Roda um sanity check estruturado no Gemini sobre a resposta candidata (pertinência e cobertura de todas as partes pedidas). Se falhar, há uma reescrita única com a mesma evidência; a segunda falha vira um fallback fixo em português. Respostas honestas de “evidência insuficiente” passam. O juiz não vê os chunks recuperados.
3. Mapeia marcadores `[[chunk-id]]` para `citations` numeradas. Marcadores desconhecidos são removidos. Se o modelo não citar, as citações ainda são montadas a partir das fontes recuperadas.

O agente não inventa conteúdo dos artigos. Se as tools não devolverem evidência, declara que não há base suficiente. O FastAPI não abre o ChromaDB nem o Gemini; as dependências (incluindo o grafo do agente, o modelo de síntese e o juiz) podem ser substituídas nos testes.

Essa separação existe para a recuperação continuar substituível e mockável, o LLM raciocinar só sobre a saída das tools, e HTTP / UI permanecerem clientes finos de `/ask`.

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

**HTTP e UI — FastAPI + Gradio.** O FastAPI concentra validação, injeção de dependências e o contrato JSON público. O Gradio (`ui.py`) é um ChatInterface que faz POST de cada mensagem em `/ask` e renderiza chips de citação; não guarda estado de retrieval nem chama o Gemini. O histórico do chat na UI é só visual.

**LLM — LangChain + Gemini 3.5 Flash.** O agente usa `langchain` `create_agent` com `langchain-google-genai` (`ChatGoogleGenerativeAI`) para o function calling nativo do Gemini acionar `search_documents` e `extract_section`. LangChain é só a camada LLM/tools — não loaders, embeddings nem o retriever do Chroma. ReAct foi rejeitado porque o desafio pede function calling nativo. O modelo padrão é `gemini-3.5-flash` (`GEMINI_MODEL`); Gemini 2.0 Flash não está disponível. Temperatura `0.0`. `ModelCallLimitMiddleware(run_limit=1)` limita as chamadas com tools; quando o limite dispara, há uma síntese sem tools.

**Citações — marcadores do modelo, API numerada, hover na UI.** O modelo deve copiar os identifiers dos chunks como `[[id]]`. O agente mapeia ids conhecidos para `[n]` mais `{number, paper_id, page, snippet}`. O Gradio envolve esses números em chips (paper id, página, trecho). Se a resposta tiver citações mas nenhum `[n]` inline, o Gradio ainda mostra uma linha **Fontes**.

**Sanidade da resposta — Gemini estruturado, sem retrieval.** Depois de uma resposta candidata, uma chamada estruturada extra verifica pertinência e completude. Há uma reescrita única com a mesma evidência das tools. O juiz é injetado como `AnswerSanityChecker` para os testes não precisarem do Gemini.

## Limitações conhecidas

- O corpus é fechado: só os três IDs obrigatórios do arXiv são ingeridos e pesquisáveis.
- As respostas dependem da extração de texto do PyPDF2; layout, figuras e equações não são recuperados como conteúdo estruturado.
- A extração de seções usa regex de headings (`abstract` / `introduction` / `conclusion`) e pode falhar em artigos com títulos atípicos.
- O agente não tem memória entre requisições. Há uma chamada de modelo com tools e, no máximo, uma síntese sem tools.
- A qualidade da recuperação é limitada pelos embeddings MiniLM, pelo tamanho do chunk e pelo `limit` da tool de busca (padrão 5, escolhido pelo modelo). Não há busca híbrida BM25 + vetorial. Se a reescrita da query para inglês falhar, a busca cai na pergunta original e o recall pode cair.
- O contrato HTTP público devolve `question`, `answer` e `citations`. Os chunks completos e os erros de tools permanecem internos. As falhas do Gemini (429 de cota, 503 de sobrecarga) devolvem HTTP `503` com o motivo original, para que um problema de cota nunca seja confundido com falta de evidência.
- `/ask` pode consumir chamadas extras ao Gemini no juiz de sanidade e em uma reescrita única depois que já existe uma resposta candidata. Se o juiz ou a reescrita esgotarem a cota ou falharem por outro motivo upstream, o endpoint devolve HTTP `503` mesmo quando já havia uma resposta candidata.
- Os números de citação só se ligam a ids de chunks de `search_documents`. A saída de `extract_section` não é citada a menos que o mesmo chunk também tenha aparecido na busca. Se o modelo omitir `[[id]]`, a API ainda devolve citações das fontes recuperadas, e o Gradio acrescenta Fontes.
- `GEMINI_API_KEY` é obrigatória na hora da requisição (`503` se estiver ausente). Importar a aplicação não abre a rede nem o banco.
