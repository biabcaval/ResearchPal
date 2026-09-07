# ResearchPal
An AI-powered Q&amp;A agent for systematic literature reviews and academic research.

## Estrutura

- `researchpal/agent`: recuperação e orquestração da pesquisa.
- `researchpal/tools`: acesso ao ChromaDB.
- `researchpal/api`: aplicação FastAPI.
- `researchpal/models`: modelos de entrada e saída.
- `researchpal/config`: configurações por ambiente.
- `researchpal/pipeline`: pipeline de download, extração e ingestão de PDFs.

## Executar

Instale as dependências com `uv sync` e ingira artigos:

```bash
python ingest.py
```

Para iniciar a API:

```bash
uvicorn app.main:app --reload
```

As configurações `RESEARCHPAL_CHROMA_PATH`, `RESEARCHPAL_COLLECTION` e
`RESEARCHPAL_PDF_DIRECTORY` podem ser definidas por variáveis de ambiente.
Os PDFs são armazenados em `data/pdfs` e o ChromaDB em `data/chroma` por padrão.
O pipeline processa exatamente os IDs `1706.03762`, `1810.04805` e `2005.11401`.

A ingestão divide cada página em chunks de 1000 caracteres, com sobreposição de
200 caracteres (`RESEARCHPAL_CHUNK_SIZE` e `RESEARCHPAL_CHUNK_OVERLAP`).
Os chunks recebem IDs determinísticos (`artigo-página-chunk`), portanto o
`upsert` é idempotente ao reexecutar a ingestão. O ChromaDB usa explicitamente
`DefaultEmbeddingFunction` para indexação e consulta. PDFs vazios, páginas sem
texto e falhas de download interrompem a ingestão com erro explícito.
