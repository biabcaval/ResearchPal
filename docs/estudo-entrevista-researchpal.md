# ResearchPal — guia de estudo para entrevista técnica

Documento em português (Brasil) para treinar a defesa deste repositório: o que o sistema faz, por que cada escolha existe, o que você diria em voz alta, e perguntas que costumam aparecer. Tudo abaixo descreve o código **como está hoje**, não um RAG “genérico de livro”.

Como usar: leia a Parte 1 em voz alta (2–5 min). Depois treine a Parte 2 (decisões). Na véspera, faça as perguntas da Parte 6 sem olhar as respostas.

---

## Parte 0 — Pitch de 60 segundos

ResearchPal é um serviço RAG **local e de corpus fechado**. Ele responde perguntas em português sobre **exatamente três papers do arXiv**:

| ID | Paper (na prática) |
|----|--------------------|
| `1706.03762` | *Attention Is All You Need* (Transformer) |
| `1810.04805` | BERT |
| `2005.11401` | Retrieval-Augmented Generation (RAG) |

O fluxo é: CLI de ingestão baixa os PDFs, extrai texto com PyPDF2, parte em chunks de tokens e grava num **ChromaDB persistente** em disco. Em runtime, o cliente manda `POST /ask` para o FastAPI. Um agente Gemini via LangChain `create_agent` (function calling nativo, no máximo 3 rodadas) decide se chama `search_documents` (busca semântica no Chroma) ou `extract_section` (regex de heading no PDF). A resposta pública é só `{question, answer}`. A UI Gradio é um **cliente HTTP fino** desse endpoint; não carrega modelo nem banco.

Frase de fechamento: *“A recuperação é determinística e testável; o LLM só raciocina sobre o que as tools devolveram.”*

---

## Parte 1 — O que é o produto e o que não é

### É

- Agente de Q&A para **revisão sistemática da literatura** sobre um conjunto **fixo** de artigos.
- Serviço HTTP (FastAPI) + índice local (Chroma) + PDFs em `data/pdfs`.
- Function calling nativo do Gemini, orquestrado por LangChain `create_agent` (não ReAct, não LlamaIndex/CrewAI).
- Testes que **não** batem na rede, no Chroma real nem no Gemini real.

### Não é

- Busca aberta na web ou no arXiv em tempo de pergunta.
- Chat com memória entre mensagens.
- Sistema de citações no contrato HTTP público (fontes existem **internamente** no `AskResponse`).
- Pipeline de Dataflow / Vertex / GCS (este repo é Python local).
- Produto com autenticação, multi-usuário ou deploy em nuvem.

Se perguntarem “é um RAG?”, responda: **sim, RAG clássico retrieve-then-generate**, mas com um passo extra: o **agente escolhe as tools**, não um retriever único sempre chamado. Há também um atalho `POST /query` que chama só a busca, sem o LLM.

---

## Parte 2 — Arquitetura (desenhe isso no quadro)

```
  ingest.py (offline, uma vez)
       arXiv PDF → data/pdfs/{id}.pdf
       PyPDF2 por página → chunks 256/32 tokens
       upsert Chroma  data/chroma  collection "papers"

  Cliente HTTP / Swagger / Gradio (ui.py)
       POST /ask  {"question": "..."}
              │
              v
       FastAPI  app.main:app
       validação Pydantic + Depends
              │
              v
       GeminiResearchAgent
       LangChain create_agent
       ChatGoogleGenerativeAI  temperature 0.0  máx. 3 rounds
              │
              ├── search_documents → VectorStore.search → Chroma
              └── extract_section  → PDF local + regex de heading
              │
              v
       AskHttpResponse {question, answer}   # fontes NÃO saem
```

### Pacotes (`src/researchpal/`)

| Pacote | Responsabilidade |
|--------|------------------|
| `config` | `Settings` (pydantic-settings), `REQUIRED_ARXIV_IDS` |
| `models` | Contratos internos: request, tools, RAG, metadados |
| `pipeline` | Download, chunk, upsert |
| `tools` | Classes determinísticas: busca e extração |
| `agent` | LangChain `create_agent` + function calling nativo |
| `api` | FastAPI, schemas HTTP, DI |
| `ui` | Gradio + cliente `requests` |

Entrada ASGI: `app/main.py` só reexporta `researchpal.api.main.app` para `uvicorn app.main:app`.

### Por que essa divisão de camadas

1. **HTTP não abre Gemini nem Chroma na importação.** `get_research_agent` e `get_vector_store` são lazy. Importar o app nos testes não precisa de API key.
2. **Tools não decidem o fluxo.** Elas recebem params validados, falam com **uma** fonte, devolvem `ToolResult`.
3. **Agente é a camada de decisão.** Expõe `StructuredTool`, executa o que o modelo pediu, sintetiza em português.
4. **UI não conhece o domínio RAG.** Só POST `/ask`. Trocar Gradio por outro cliente não mexe no índice.

---

## Parte 3 — Conceitos RAG aplicados a *este* código

Não recitar Wikipedia. Amarre cada conceito a um símbolo do repo.

### 3.1 Corpus fechado

`REQUIRED_ARXIV_IDS` é uma tupla **hard-coded**. `ingest_papers` recusa qualquer outro conjunto (`ValueError`). `extract_section` recusa `paper_id` fora da lista (`ToolResult` de falha, sem exception).

**Por quê:** o enunciado do projeto é revisão sobre um conjunto fixo. Corpus aberto exigiria crawler, rate limit, atualização de índice e avaliação contínua. Aqui a avaliação é reproduzível: três PDFs, IDs determinísticos.

### 3.2 Ingestão vs query time

| Momento | O que acontece |
|---------|----------------|
| Ingestão (`ingest.py`) | Rede só para baixar PDF; embedding local MiniLM inglês; grava disco |
| Query (`/ask`) | Reescreve a pergunta para inglês; embed no **mesmo** encoder; Gemini na nuvem |

Consistência de embedding: o mesmo `paper_embedding_function()` (ONNX `all-MiniLM-L6-v2`) na criação da collection e na query. A comparação é inglês-com-inglês: `search_documents` chama `to_english_retrieval_query` antes do Chroma. Se a collection antiga for multilingual, o ingest reconstrói (`rebuild_incompatible_collection=True`).

### 3.3 Chunking por tokens com overlap

`chunk_text` em `pipeline/ingestion.py`:

- Encoding `tiktoken` **`cl100k_base`** (`TOKEN_ENCODING`)
- `chunk_size` padrão **256** tokens, `overlap` **32**
- `step = chunk_size - overlap` → **224**
- encode → janela nos ids → decode; texto já `strip()`
- vazio → lista vazia
- overlap ≥ size → `ValueError` (também validado em `Settings`)

Chunking é **por página**, não no PDF inteiro. Metadado `page` fica em todo chunk → citação possível (mesmo que a API pública não a exponha).

**Por que tokens e não caracteres:** MiniLM consome tokens (~256). 1000 caracteres ≠ 1000 tokens, então a janela antiga podia estourar ou subutilizar o encoder. `tiktoken` é determinístico e padrão em RAG. Não é o tokenizer nativo do MiniLM nem do Gemini — é um orçamento reproduzível.

**Por que por página e não documento inteiro:** PyPDF2 já entrega páginas; page number é o melhor “locator” barato para paper acadêmico.

### 3.4 IDs determinísticos e upsert idempotente

Formato: `{paper_id}-page-{page_number}-chunk-{chunk_index}`

- `page_number` começa em **1**
- `chunk_index` começa em **0**

Reexecutar ingestão **sobrescreve** os mesmos IDs (`collection.upsert`). Não duplica o índice se o algoritmo de chunk não mudar.

Se mudar `chunk_size`, os IDs mudam (porque o índice do chunk muda) — chunks velhos órfãos podem ficar na collection. Ponto honesto se perguntarem “é realmente idempotente em qualquer config?”: **idempotente para a mesma config de chunking**.

### 3.5 Download atômico

Arquivo temporário `{id}.pdf.part` → `replace` no destino. Valida `content-type == application/pdf` e tamanho ≠ 0. Se o PDF local já existe e não está vazio, **não baixa de novo**. Falha → `RuntimeError` com log, apaga o `.part`.

### 3.6 Embeddings MiniLM inglês (não Gemini)

`paper_embedding_function()` → Chroma `DefaultEmbeddingFunction` (ONNX `all-MiniLM-L6-v2`). Offline depois do download do modelo ONNX. Sem cota Google para vetores. Os papers ficam em inglês. `search_documents` reescreve a query para inglês (Gemini) e só então compara. A resposta gerada é sempre em português. Se a reescrita falhar, cai na query original. Troca de encoder: ingest reconstrói a collection; no query path, mismatch vira 503 pedindo `ingest.py`.

### 3.7 Retrieval

`search_documents` chama `to_english_retrieval_query` e só então `VectorStore.search` → `collection.query(query_texts=[english_query], n_results=limit)`. Perguntas em português e em inglês viram query de busca em inglês. A resposta ao usuário permanece em português.

- Default de `limit` na tool: `SearchToolParams.limit = 5` (1–100).
- Distância do Chroma entra em `RetrievedDocument.distance` (pode ser `None` se o payload vier incompleto).
- **Não há** híbrido BM25 + vetor, **não há** rerank, **não há** filtro por `retrieval_score_threshold`.

Armadilha do código: `Settings.retrieval_limit` e `retrieval_score_threshold` **existem no `.env` mas não estão ligados** à busca. O limite real vem do argumento da tool (o modelo pode pedir `limit`). Se o entrevistador ler o README, o README ainda fala do `RESEARCHPAL_RETRIEVAL_LIMIT` como bound de qualidade — no código da query, o bound operacional é o `limit` da tool.

### 3.8 Extração de seção (não é RAG vetorial)

`extract_section` concatena todas as páginas e varre linhas com regex:

- abstract: linha `abstract` opcionalmente com `:`
- introduction / conclusion: heading numerado opcional (`1 Introduction`, `5 Conclusions`)

Corta no próximo heading conhecido. Se não achar → `ToolResult` de erro, não alucinação.

Por que duas tools: busca semântica acha trechos no meio do paper; seção é o que o usuário pede quando fala “qual o abstract do BERT?”. Regex é frágil em heading atípico — limitação documentada.

### 3.9 Generate: Gemini Flash + function calling

- LangChain `create_agent` + `ChatGoogleGenerativeAI` (`langchain-google-genai`).
- Modelo default `gemini-3.5-flash` (`GEMINI_MODEL`).
- `temperature=0.0` nas chamadas com tools **e** na síntese forçada.
- Function calling nativo: schemas das tools = Pydantic (`SearchToolParams` / `ExtractSectionParams`) via `as_langchain_tool()`. Não ReAct.

System instruction (inglês, resposta em português):

- usar **somente** evidência das tools
- `search_documents` deve receber a query **em inglês** (comparação com os papers originais)
- nunca inventar fatos, citações, conteúdo de seção
- se tools falharem ou vierem vazias, declarar falta de evidência
- mencionar IDs dos papers quando possível

Descrições das tools LangChain estão em **português**. A de `search_documents` pede query em inglês.

### 3.10 Loop de tools (o coração da entrevista)

`GeminiResearchAgent.ask`:

1. Invoca o grafo LangChain com `HumanMessage(pergunta)`.
2. O modelo pode emitir `tool_calls` nativos. LangChain executa o `StructuredTool`; o adapter devolve JSON de `ToolResult`.
3. `ModelCallLimitMiddleware(run_limit=3)` impede a 4ª chamada com tools. A mensagem artificial “Model call limits exceeded…” **não** vai ao usuário.
4. Sem tool calls → texto do último `AIMessage` vira resposta.
5. Se o limite estourou → `_synthesize_without_tools` no `ChatGoogleGenerativeAI` **sem** tools, prompt PT *“Não chame mais nenhuma ferramenta…”*. Se houver texto, `tool_errors` inclui `"Maximum tool-calling rounds exceeded"`.

Dedup de fontes: `_unique_sources` por `identifier`.

`evidence_found = bool(sources or sections)` — evidência = tool **sucesso com dados**, não “o modelo falou que sim”.

---

## Parte 4 — Decisões de implementação (formato “por que não X”)

Treine cada bloco no formato: **escolha → alternativa rejeitada → trade-off → onde está no código**.

### D1. Chroma persistente, não Postgres/pgvector, não Pinecone, não FAISS solto

- Corpus de 3 papers, um desenvolvedor, máquina local.
- `PersistentClient(path=data/chroma)` evita operar um banco.
- FAISS exigiria serializar metadados à parte; Pinecone exigiria conta e rede na query.
- Limitação: não é multi-processo sofisticado nem replica; serve o escopo.

### D2. LangChain no agente, não no RAG

- Escolha: `create_agent` + `ChatGoogleGenerativeAI` para falar com o Gemini.
- Alternativa rejeitada: loop manual `google-genai` (já funcionava) e ReAct (o spec pede function calling nativo).
- LangChain não entra em ingestão, embeddings MiniLM, nem Chroma. Tools continuam classes com `ToolResult`.
- Teto de 3 rounds: `ModelCallLimitMiddleware`, depois síntese sem tools — o limite não some dentro do framework.
- Testes injetam `graph` e `synthesis_model`; não batem no Gemini real.

### D3. Tools devolvem `ToolResult`, não levantam erro de negócio

Falhas de busca/PDF/seção viram `success=False, error=str`. O modelo **vê** o erro no function response e pode tentar outra tool ou confessar falta de evidência.

Exceções de infraestrutura do **Gemini** são outra classe: `ModelUnavailableError` → HTTP **503**. Cota 429 não pode parecer “não achei no corpus”.

Essa distinção é a decisão de produto mais “sênior” do repo. Ensaie:

> 200 com texto “não há evidência” = o sistema funcionou e o corpus não cobre.  
> 503 = o sistema não conseguiu falar com o modelo.  
> 422 = o cliente mandou JSON inválido (`extra="forbid"`).

### D4. Contrato HTTP estreito

Interno `AskResponse`: `answer`, `sources`, `sections`, `evidence_found`, `tool_errors`.  
Público `AskHttpResponse`: só `question` e `answer`.

Por quê: UI simples; não vazar erros de tool e IDs internos por padrão. Custo: o usuário não vê citações. `/query` existe para debug/busca crua e **devolve** `ToolResult` completo.

### D5. FastAPI Depends + override nos testes

`TestClient` injeta agente/store fake. Prova que a rota não precisa do Google. `get_research_agent` mapeia falha de init (key ausente, Chroma, OSError) para 503.

### D6. Pydantic em todas as fronteiras

| Fronteira | Modelo | extra |
|-----------|--------|--------|
| HTTP `/ask` | `AskRequest` | forbid |
| HTTP resposta | `AskHttpResponse` | forbid |
| Params de tool | `SearchToolParams`, `ExtractSectionParams` | forbid |
| Payload Chroma | `ChromaQueryResult` | **ignore** + `None` → `[]` |
| Metadado chunk | `ChunkMetadata` | ignore; `to_chroma()` **remove nulls** (Chroma não aceita null) |
| LangChain messages + `ToolResult` JSON | `HumanMessage` / `AIMessage` / `ToolMessage` + `ToolResult[T]` | `StructuredTool` via `as_langchain_tool()` |

Se metadado do Chroma for inválido: fallback `paper_id=identifier, page=1` + warning — degradação, não crash na busca.

### D7. PyPDF2, não Unstructured / Marker / OCR

Simples, já no `pyproject`. Não reconstrói figuras, layout, equações. Páginas sem texto viram `""` em `read_pdf_pages`; se **o paper inteiro** não tem texto, ingestão aborta.

### D8. uv + `pyproject.toml`, Python ≥ 3.12

Sem `requirements.txt`. `uv sync --dev`. Runtime: chromadb, google-genai, langchain, langchain-google-genai, fastapi, pypdf2, pydantic, pydantic-settings, requests, uvicorn, gradio. Dev: pytest.

### D9. Gradio em processo separado

Spec: `docs/superpowers/specs/2026-09-08-gradio-ui-design.md`.

- `respond(message, history)` **ignora** `history`.
- Timeout da UI `RESEARCHPAL_ASK_TIMEOUT` default **180s** (agente + até 3 rounds); timeout de download arXiv é **30s**.
- Erros: pergunta vazia sem HTTP; timeout; connection (“Is uvicorn running?”); não-200 com `detail` do FastAPI.

Não montar Gradio dentro do FastAPI: dois processos, dois ports, UI não segura o event loop da API.

### D10. Temperatura 0 e Flash

Corpus pequeno, tarefa factual. Flash = latência/custo. 0.0 reduz variação (não zera alucinação: o prompt + tools são a defesa real).

---

## Parte 5 — Caminhos de erro (desenhe a matriz)

| Situação | Camada | Resultado |
|----------|--------|-----------|
| JSON com campo extra em `/ask` | Pydantic | **422** |
| `question` vazio | Pydantic min_length=1 | **422** |
| Sem `GEMINI_API_KEY` | init do agente / Depends | **503** |
| Gemini 429 / 503 / APIError | `ModelUnavailableError` | **503** + `detail` |
| Chroma fora no `/query` | Depends store | **503** |
| Tool search falha | `ToolResult` | modelo vê erro; `/ask` ainda **200** se o modelo responder |
| Tools sem dados | `_no_evidence_response` | **200**, texto PT padrão |
| Texto vazio do Gemini após tools | no-evidence ou “recuperei evidências mas não sintetizei” | **200** |
| 3 rounds | síntese sem tools | **200** + erro interno de max rounds |
| `paper_id` inválido em extract | ToolResult | não levanta |
| PDF ausente | ToolResult | não levanta |
| Download arXiv falhou | ingestão | `RuntimeError`, CLI morre |
| Overlap ≥ chunk_size | Settings | `ValueError` no boot de config |

Mensagens PT que vale decorar:

- Sem evidência: *“Não encontrei evidência suficiente nos artigos indexados para responder com segurança.”*
- Evidência coletada mas síntese vazia: *“Recuperei evidências nos artigos, mas não consegui concluir uma síntese final automaticamente. Consulte as fontes retornadas.”* (as fontes **não** vão no HTTP público — nuance: a mensagem faz mais sentido no modelo interno.)

---

## Parte 6 — Banco de perguntas (com gabarito curto)

Responda em voz alta, depois confira. As marcadas com ★ costumam ser as primeiras.

### A. Produto e escopo

**A1. ★ O que o ResearchPal faz?**  
Q&A em português sobre três papers fixos do arXiv, via RAG local + agente Gemini com tools.

**A2. Por que só três papers?**  
Escopo de revisão sistemática controlada; ingestão validada; testes reproduzíveis. IDs: Transformer, BERT, RAG — o corpus *é* o tema.

**A3. O usuário pode perguntar sobre um quarto paper?**  
Ingestão recusa outros IDs. Busca só o que está no Chroma. `extract_section` recusa `paper_id` fora da whitelist. O modelo pode até tentar; a tool falha e o prompt manda confessar falta de evidência.

**A4. Tem memória de conversa?**  
Não. API stateless. Gradio mostra histórico só na UI; cada mensagem é um `/ask` independente.

### B. RAG e retrieval

**B1. ★ Explique o pipeline de ingestão ponta a ponta.**  
Valida IDs → download streaming atômico (skip se PDF existe) → PyPDF2 por página → `chunk_text` 256/32 tokens (`cl100k_base`) → IDs `{paper}-page-{n}-chunk-{i}` → `VectorStore.upsert` na collection `papers`.

**B2. ★ Por que chunk de tokens e não de caracteres?**  
MiniLM orça contexto em tokens. `tiktoken` `cl100k_base` é determinístico e o default de RAG. Trade-off: não é o tokenizer do MiniLM/Gemini; ainda pode cortar no meio de uma frase.

**B3. Por que overlap 32?**  
Frases e headings que caem no corte aparecem nos dois chunks vizinhos. ~12.5% de 256; não foi tunado com eval neste repo.

**B4. Como você citaria a página na resposta?**  
Metadado `page` no chunk. Hoje o HTTP não devolve `sources`. Internamente o agente tem a lista. Melhoria natural: incluir fontes no contrato ou no prompt pedindo para citar `paper_id` + página.

**B5. MiniLM vs embeddings Gemini?**  
MiniLM inglês (ONNX): offline, barato, consistente ingest/query. A cota do Gemini entra na geração e na reescrita PT→EN da query de busca (não na ingestão). Gemini embeddings: melhor qualidade, custo na ingestão.

**B6. O que é “distance” no resultado?**  
Métrica do Chroma para aquele embedding (não é similaridade cosseno já invertida — não assuma o nome sem olhar a versão do Chroma). Não há corte por threshold no código.

**B7. Como faria busca híbrida?**  
BM25 (lexical: nomes próprios, IDs, termos raros) + vetor; fundir (RRF) e talvez rerank. Não está implementado. Bom upgrade se a reescrita falhar em keywords.

**B8. Chunking por página vs sliding window no paper inteiro?**  
Página preserva locator e limita contexto de layout. Sliding no doc inteiro evitaria corte no fim da página no meio de um parágrafo que atravessa página — PyPDF2 não junta isso.

**B9. `retrieval_limit` no Settings funciona?**  
Campo existe; a busca usa `SearchToolParams.limit`. Resposta honesta: config morta / dívida. Ligaria ou removeria.

**B10. Por que upsert e não add?**  
Reingest idempotente com os mesmos IDs. `add` duplicaria ou falharia em ID existente.

### C. Agente e LLM

**C1. ★ Qual a diferença entre tool e agente neste repo?**  
Tool: determinística, sem prosa, uma fonte, `ToolResult`. Agente: Gemini escolhe tools, loop, síntese PT, nunca deveria inventar paper.

**C2. Por que no máximo 3 rounds?**  
`ModelCallLimitMiddleware(run_limit=3)` limita chamadas com tools; depois `_synthesize_without_tools` no `ChatGoogleGenerativeAI` sem tools.

**C3. O que acontece se o modelo pedir uma tool que não existe?**  
`_execute_tool` devolve `Unknown tool: ...` como `ToolResult` falho; o modelo recebe isso.

**C4. Por que temperature 0?**  
Tarefa factual, reproduzibilidade. Não substitui grounding.

**C5. System prompt em inglês e resposta em português — por quê?**  
Instrução estável (modelos frequentemente raciocinam melhor nas rules em EN) + produto PT. Descriptions das tools em PT alinham com a pergunta do usuário.

**C6. Como o SDK é isolado nos testes?**  
Injeção de `graph` e `synthesis_model` no `GeminiResearchAgent`. Testes usam fakes que implementam `invoke`, sem chamar Gemini nem LangChain reais.

**C7. Function calling vs “prompt: chame JSON”?**  
JSON solto quebra, não valida schema, difícil de testar. Function calling nativo + Pydantic `model_validate` nos args.

**C8. O agente pode alucinar mesmo com as tools?**  
Sim, se ignorar o prompt e bordar sobre os chunks. Mitigações: temperature 0, evidência only, max rounds, testes de “tools vazias → texto de falta de evidência”. Não há fact-check automático contra o PDF depois da síntese.

**C9. Por que não RAG “sempre retrieve depois generate” sem agente?**  
`/query` é esse caminho. `/ask` precisa **escolher** extract vs search e eventualmente ambos. Function calling justifica o agente.

**C10. Flash vs Pro?**  
Default Flash. Pro: melhor tool use, mais caro. `GEMINI_MODEL` já permite trocar sem código.

### D. API, HTTP e erros

**D1. ★ 503 vs “não há evidência”.**  
Infra/quota/key/Chroma init → 503. Corpus não cobre / tools vazias → 200 + frase PT. Comentário no `api/main.py`: quota não pode mascarar falta de evidência.

**D2. Endpoints?**  
`GET /health` → `{status: ok}` (sem checar Gemini/Chroma).  
`POST /query` → busca.  
`POST /ask` → agente.

**D3. `/health` é health de verdade?**  
É liveness do processo, não readiness do índice. Entrevistador forte vai puxar isso: health “ok” com Chroma vazio e sem key (key só explode no `/ask`).

**D4. Por que `extra="forbid"` no body?**  
Contrato rígido; clientes não mandam campos silenciosamente ignorados. Teste: `test_ask_endpoint_rejects_extra_fields`.

**D5. Lazy init, por quê?**  
`uvicorn --reload` e pytest importam o módulo. Abrir Chroma/Gemini no import acopla boot a I/O e quebra testes.

**D6. UI timeout 180s vs download 30s.**  
Ask envolve vários RTTs Gemini + busca. Download é um GET de PDF.

### E. Dados, PDF, regex

**E1. Como a seção é achada?**  
Scan linha a linha; start após heading; end no próximo heading do mapa `SECTION_HEADINGS`.

**E2. E se o paper usar “Concluding remarks”?**  
Regex não casa → tool error → agente deve dizer que não extraiu.

**E3. Figuras e tabelas?**  
PyPDF2 `extract_text`; figuras/equações mal recuperadas. Limitação explícita do README.

**E4. Por que validar content-type no download?**  
arXiv às vezes devolve HTML de erro/captcha. Evita indexar lixo.

### F. Testes e qualidade

**F1. ★ Como testa sem pagar Gemini?**  
Fake client no agente; `dependency_overrides` na API; `FakeCollection` no VectorStore; `requests.post` patchado na UI; PDF mínimo em bytes no teste de PyPDF2; ingestão com download/read mockados.

**F2. Qual teste você mostraria primeiro?**  
`test_ask_raises_model_unavailable_when_gemini_rejects_the_request` e o de `/ask` 503 — provam a distinção de produto. Ou o teste do agente que executa function call e sintetiza.

**F3. O que o teste da UI prova sobre memória?**  
`respond` descarta history; settings (URL/timeout) são passados ao client.

**F4. Coverage de metadado inválido?**  
`test_search_falls_back_when_metadata_is_invalid`.

### G. Design, trade-offs, “o que você faria depois”

**G1. ★ Se tivesse mais uma semana, o que implementaria?**  
Priorize o que o README já admite: (1) expor fontes no `/ask` ou na UI; (2) ligar ou remover `retrieval_limit`/`score_threshold`; (3) híbrido ou rerank se o MiniLM sofrer; (4) eval dourado (perguntas + trechos esperados); (5) readiness check no `/health`; (6) headings mais robustos.

**G2. Como avaliaria qualidade RAG?**  
Conjunto de perguntas com paper_id + página esperada; hit@k no retriever separado da nota da geração; recusa em perguntas off-corpus; teste de 429.

**G3. Como escalaria para 10k papers?**  
Chroma embarcado deixa de ser óbvio: lote de ingestão, embeddings em batch, talvez pgvector/Vertex AI Vector Search, chunking melhor, OCR, filas, observabilidade. O agente de 3 rounds ainda serve; o gargalo vira retrieval e ingestão.

**G4. Segurança?**  
Sem auth. Qualquer um no localhost chama `/ask` (custa a chave). Path de PDF é `{pdf_directory}/{paper_id}.pdf` com whitelist — reduz path traversal. Não logar a API key. `.env` fora do git.

**G5. Por que não async FastAPI + client async do Gemini?**  
Escopo sincrono, código linear, testes simples. `ask_timeout` 180s segura o cliente. Upgrade se houvesse concorrência real.

**G6. Por que collection name `papers`?**  
Um corpus, uma collection. Multi-corpus exigiria nome por revisão / tenant.

**G7. `get_settings()` cria Settings novo sempre.**  
Não é singleton cacheado. Cada chamada relê env. Simples; em teoria poderia divergir se env mudar no processo — irrelevante aqui.

**G8. Por que Protocol no VectorStore?**  
Injeção de fake store sem herdar Chroma. Testes de tools e agente estáveis sem disco nem embedding real.

### H. Perguntas “pegadinhas” e respostas honestas

**H1. O README diz que o limite de retrieval é `RESEARCHPAL_RETRIEVAL_LIMIT`. Confere?**  
O default 5 coincide com o default da tool, mas o settings field não é lido em `search()`. Fale isso; mostra que você leu o código.

**H2. A mensagem “consulte as fontes retornadas” chega no cliente HTTP?**  
O texto sim, se esse ramo disparar; as fontes não. Pequena inconsistência de UX.

**H3. `/query` não usa o agente. Isso é RAG?**  
É só retrieve. Não passa pelo loop de tools. Ainda pode gastar Gemini para reescrever a query em inglês antes do MiniLM.

**H4. O schema das tools vem de onde?**  
Do Pydantic `params_model` de cada `ResearchTool`, exposto ao modelo via `as_langchain_tool()` como `args_schema` do `StructuredTool`.

**H5. Ingestão e API precisam do mesmo `chroma_path`.**  
Sim. Se a API apontar para outro diretório, busca vazia (e o modelo dirá falta de evidência — 200, não 503).

**H6. Embeddings são em inglês e a pergunta em português.**  
Sim, e por isso a query é reescrita para inglês antes do MiniLM. A comparação é EN×EN. A resposta final é em português.

**H7. `health` não prova que o índice tem os 3 papers.**  
Correto. Um check de readiness contaria IDs ou um ping na collection.

**H8. Por que `requests` e não `httpx`?**  
Já usado no download; UI reutiliza. Sem async.

---

## Parte 7 — Roteiros de entrevista (simule o relógio)

### 2 minutos (elevator)

ResearchPal é RAG local de corpus fechado. Três PDFs do arXiv viram chunks no Chroma com MiniLM inglês. FastAPI recebe a pergunta; a query de busca é reescrita para inglês; Gemini Flash com function calling chama busca semântica ou extração de seção no PDF; responde só com essa evidência, em português. UI Gradio só faz HTTP. Tools testáveis; falha do Gemini é 503, falta de evidência é 200.

### 5 minutos (arquitetura)

1. Desenhe ingestão vs runtime.  
2. Tools vs agente.  
3. Contrato interno vs HTTP.  
4. 3 rounds + síntese forçada.  
5. Uma limitação: MiniLM + regex + sem fontes na API.

### 15 minutos (deep dive)

Inclua: IDs determinísticos, `to_chroma()` sem null, Depends lazy, testes fake (`graph` / `synthesis_model`), matriz 422/200/503, dívida do `retrieval_limit`.

### Whiteboard: “o usuário pergunta o abstract do BERT”

1. `/ask` valida `AskRequest`.  
2. Agente manda a pergunta + tools.  
3. Modelo deve chamar `extract_section(paper_id=1810.04805, section=abstract)` (validator lowercasing).  
4. Tool lê `data/pdfs/1810.04805.pdf`, regex Abstract, devolve texto.  
5. Modelo sintetiza PT.  
6. HTTP: só a prosa.  
Se o heading falhar, tool error; modelo pode cair para `search_documents("BERT abstract")`.

### Whiteboard: “Gemini está em quota”

`generate_content` levanta `APIError` → `ModelUnavailableError` → FastAPI 503 com a mensagem original. UI mostra `API error 503: ...`. **Não** vira “não achei evidência”. Testes: `test_agent_upstream_errors` e `test_api_upstream_errors`.

---

## Parte 8 — Glossário rápido (use os nomes do código)

| Termo | Nome no código |
|-------|----------------|
| Agente | `GeminiResearchAgent` |
| Falha de modelo | `ModelUnavailableError` |
| Busca | `search_documents` / `VectorStore.search` |
| Seção | `extract_section` / `_find_section` |
| Chunk | `chunk_text` + `ChunkMetadata` |
| Resultado de tool | `ToolResult[T]` |
| Resposta interna | `AskResponse` |
| Resposta HTTP | `AskHttpResponse` |
| Collection | `"papers"` |
| IDs arXiv | `REQUIRED_ARXIV_IDS` |
| Loop | `MAX_TOOL_ROUNDS = 3` |
| Cliente UI | `ask_http` |
| App ASGI | `app.main:app` |

---

## Parte 9 — Limitações (decore para parecer dono do projeto)

1. Corpus fechado de 3 IDs.  
2. PyPDF2: sem estrutura de figura/equação.  
3. Regex de seção frágil.  
4. Sem memória entre requests.  
5. MiniLM + `limit` da tool; sem híbrido; threshold de score não usado.  
6. HTTP sem fontes/tool_errors.  
7. Agente para após 3 rounds.  
8. Health não valida índice nem Gemini.  
9. Reescrita PT→EN da query (falha cai no original).  
10. Sem eval automatizado de qualidade de resposta (só testes de contrato e fakes).

---

## Parte 10 — Checklist na véspera

- [ ] Desenhar o diagrama de memória sem olhar o README.  
- [ ] Recitar os três arXiv IDs e o que cada paper é.  
- [ ] Explicar 503 vs 200-sem-evidência com um exemplo de 429.  
- [ ] Explicar ID de chunk e por que upsert é idempotente.  
- [ ] Explicar por que a UI ignora o histórico.  
- [ ] Admitir `retrieval_limit` morto e a API sem fontes.  
- [ ] Citar stack: Python 3.12, uv, FastAPI, Chroma, LangChain, google-genai, PyPDF2, Pydantic v2, Gradio, pytest.  
- [ ] Dizer uma melhoria e por que ela é a primeira (fontes no `/ask` ou eval).

---

## Parte 11 — Mapa mental de arquivos (para abrir na hora)

```
ingest.py                         CLI ingestão
ui.py                             CLI Gradio
app/main.py                       ASGI
src/researchpal/config/settings.py
src/researchpal/models/rag.py
src/researchpal/models/documents.py
src/researchpal/pipeline/ingestion.py
src/researchpal/tools/pdf.py
src/researchpal/tools/vector_store.py
src/researchpal/tools/base.py
src/researchpal/tools/search_documents.py
src/researchpal/tools/extract_section.py
src/researchpal/agent/gemini_agent.py
src/researchpal/agent/errors.py
src/researchpal/api/main.py
src/researchpal/api/dependencies.py
src/researchpal/api/schemas.py
src/researchpal/ui/app.py
src/researchpal/ui/client.py
tests/                            11 módulos, ~33 testes
docs/superpowers/specs/2026-09-08-gradio-ui-design.md
```

---

## Parte 12 — Respostas “modelo” (parágrafos prontos)

**“Por que LangChain e não o SDK Gemini direto?”**  
O desafio permite LangChain, LlamaIndex ou o SDK. LangChain `create_agent` usa function calling nativo do Gemini, então as tools continuam Pydantic + `ToolResult`. Não usei ReAct. Ingestão e Chroma ficam fora do framework. O teto de 3 rounds é `ModelCallLimitMiddleware` mais uma síntese sem tools, o mesmo contrato de antes.

**“Isso é production-ready?”**  
É um serviço local de demo/estudo: sem auth, sem eval, health raso, embeddings pequenos, contrato sem citações. As decisões (lazy DI, 503 vs evidência, tools determinísticas, ingestão fail-fast) são o que eu levaria para produção; o restante precisaria de observabilidade, readiness, fontes na API e avaliação.

**“Como você evita alucinação?”**  
Não evito 100%. Reduzo: tools como única fonte, prompt explícito, temperatura 0, recusa padronizada, extração de seção por regex em vez de pedir ao modelo “lembre o abstract”, e testes que travam o comportamento de tools vazias. O próximo passo seria citar obrigatoriamente `paper_id`+página e recusar se o retriever não bater.

---

*Gerado a partir do estado do repositório ResearchPal (pacote `researchpal` 0.1.0). Se o código mudar, priorize `gemini_agent.py`, `ingestion.py`, `api/main.py` e o README.*
