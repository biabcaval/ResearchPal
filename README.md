# ResearchPal

Agente de perguntas e respostas para revisão sistemática de literatura.

## Estrutura

- `researchpal/agent`: recuperação e orquestração da pesquisa.
- `researchpal/tools`: acesso ao ChromaDB e extração de PDFs.
- `researchpal/api`: aplicação FastAPI e dependências HTTP.
- `researchpal/models`: modelos internos do agente e das tools.
- `researchpal/config`: configurações por ambiente.
- `researchpal/pipeline`: download, extração e ingestão de PDFs.

## Arquitetura

```text
Cliente HTTP
    |
    v
FastAPI (/ask) -- validação e serialização HTTP
    |
    v
GeminiResearchAgent -- decide quando usar cada tool e sintetiza a resposta
    |                         |
    v                         v
search_documents        extract_section
    |                         |
    v                         v
ChromaDB                 PDFs locais
```

Uma **tool** executa uma operação determinística e retorna evidência estruturada:
`search_documents` consulta ChromaDB e `extract_section` lê uma seção permitida de
um PDF. O **agente** escolhe as tools via function calling, acumula evidências e
pede ao Gemini uma síntese em português. A camada HTTP não conhece ChromaDB nem
cria clientes externos diretamente; suas dependências podem ser substituídas nos
testes por `app.dependency_overrides`.

## Setup do zero

Requisitos: Python 3.12+, `uv` e uma chave do Google AI Studio.

```bash
uv sync --dev
Copy-Item .env.example .env
# Edite .env e preencha GEMINI_API_KEY
uv run python ingest.py
uv run uvicorn app.main:app --reload
```

No Linux/macOS, use `cp .env.example .env` no lugar de `Copy-Item`. O entrypoint
ASGI correto é `app.main:app`; o script de ingestão correto é `ingest.py` na raiz.
A documentação interativa fica em `http://127.0.0.1:8000/docs`.

## API

`POST /ask` aceita somente:

```json
{"question": "O que os artigos dizem sobre atenção?"}
```

e retorna:

```json
{
  "question": "O que os artigos dizem sobre atenção?",
  "answer": "..."
}
```

Campos desconhecidos ou perguntas vazias retornam `422`. Sem `GEMINI_API_KEY`,
a inicialização preguiçosa do agente retorna `503`; importar a aplicação não faz
chamadas de rede ou abre o banco. O Swagger está disponível em `/docs`.

## Execução e ingestão

O pipeline processa exatamente os IDs `1706.03762`, `1810.04805` e `2005.11401`.
Os PDFs são armazenados em `data/pdfs` e o ChromaDB em `data/chroma` por padrão.
A ingestão divide cada página em chunks de 1000 caracteres, com sobreposição de
200 caracteres (`RESEARCHPAL_CHUNK_SIZE` e `RESEARCHPAL_CHUNK_OVERLAP`).
Os chunks recebem IDs determinísticos (`artigo-página-chunk`), portanto o
`upsert` é idempotente. PDFs vazios, páginas sem texto e falhas de download
interrompem a ingestão com erro explícito.

## Decisões e limitações

O agente usa a biblioteca oficial `google-genai` para function calling e configura
o modelo por `GEMINI_MODEL`. O agente não mantém memória entre requisições e
limita o ciclo de function calling a três rodadas. O contrato HTTP é menor que
o modelo interno do agente: fontes e erros continuam disponíveis internamente
sem acoplar clientes ao formato de implementação.

A recuperação depende dos PDFs locais já ingeridos, e a qualidade da resposta
depende do modelo Gemini configurado e da extração textual do PDF. Configure
`GEMINI_API_KEY` somente no arquivo `.env`; nenhuma chave é armazenada no código
ou no repositório.

## Testes

Os testes não fazem chamadas externas: ChromaDB, PDFs, rede e cliente Gemini são
substituídos por mocks.

```bash
uv run pytest
```
