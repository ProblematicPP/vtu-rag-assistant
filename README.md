# VTU RAG Assistant

An agentic retrieval-augmented generation (RAG) assistant for **VTU (Visvesvaraya Technological University)**
engineering students. Put your module notes in a folder, and it answers exam questions using only those
notes, citing the **subject → module → note (→ page)** each answer came from.

Everything runs locally with Docker Compose: the LLM runs on your GPU through Ollama.

```
┌──────────┐   ┌──────────────────────────── FastAPI ─────────────────────────────┐
│  Gradio  │──▶│ /ask ──────────────▶ hybrid search ─▶ answer with [n] citations  │
│ Telegram │   │ /agentic-ask ─▶ LangGraph:                                        │
└──────────┘   │   guardrail ─▶ retrieve ─▶ grade ─┬─▶ generate                    │
               │                  ▲                └─▶ rewrite query ─┐            │
               │                  └───────────────────────────────────┘            │
               │ /notes, /notes/sync ─▶ parse ─▶ section chunks ─▶ Jina embed ─▶ index │
               └──────┬────────────┬──────────────┬─────────────┬─────────────┬─────┘
                  PostgreSQL   OpenSearch       Redis        Ollama       Langfuse
                  (metadata)  (BM25 + k-NN)    (cache)     (GPU LLM)    (tracing, optional)
                                   ▲
                 Airflow DAG ──────┘  scheduled re-index of new/changed notes
```

## Quick start

Prerequisites: Docker Desktop (with WSL2 GPU support on Windows) or Docker Engine plus the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/),
and about 12 GB of free RAM for the full stack.

```bash
cp .env.example .env          # optional: add JINA_API_KEY for hybrid (vector) search
docker compose up --build
```

On the first start, `ollama-init` downloads the model (`llama3.2:3b`, about 2 GB). The API indexes
everything under `data/` in the background as it starts (`SYNC_ON_STARTUP`), which includes the sample
note, so you can ask questions right away.

| Service | URL |
|---|---|
| Chat UI (Gradio) | http://localhost:7860 |
| API docs (Swagger) | http://localhost:8000/docs |
| Health | http://localhost:8000/health |
| Airflow (`admin` / `admin`) | http://localhost:8080 |
| Langfuse (`admin@example.com` / `vtu-rag-admin`) | http://localhost:3000 |
| OpenSearch | http://localhost:9200 |

**No NVIDIA GPU?** Use `docker compose -f docker-compose.yml -f docker-compose.cpu.yml up --build`.

**Answer quality.** The default `llama3.2:3b` fits in 6 GB of VRAM and answers in a few seconds (the
first request after startup is slower while the model loads). Being small, it sometimes adds detail
that isn't in your notes, even though the prompt forbids it — always check answers against the cited
excerpt. With more VRAM, set `OLLAMA_MODEL=qwen2.5:7b-instruct` (or another larger model) in `.env`
for noticeably better grounding, then run `docker compose run --rm ollama-init`.

**Port already in use?** Every host port is configurable in `.env` (`API_HOST_PORT`, `REDIS_HOST_PORT`, and so on).

## Adding notes

Put files under `data/` using this layout (see [data/README.md](data/README.md)):

```
data/<branch>/<scheme>/sem<N>/<SUBJECT_CODE>/module<N>[-anything].<pdf|md|txt>
data/cse/2022/sem3/BCS303/module2.pdf
```

Then index them in any of these ways:

- CLI: `docker compose exec api python scripts/ingest.py` (add `--force` to re-index everything)
- API: `POST /api/v1/notes/sync`
- Airflow: the `reindex_vtu_notes` DAG runs every 30 minutes (`REINDEX_SCHEDULE`)
- Upload: `POST /api/v1/notes` (multipart) stores the file in the right folder and indexes it:

```bash
curl -F file=@os-module2.pdf -F subject_code=BCS303 -F module_number=2 \
     http://localhost:8000/api/v1/notes
```

Unchanged notes are skipped by content hash. If you add a `JINA_API_KEY` later, notes that were indexed as
BM25-only get embedded on the next sync.

### Scanned notes

Most VTU notes in circulation are photocopies scanned to PDF — pages of images with no text to extract.
Those are handled automatically: when too many pages come back near-empty, the file goes through
[ocrmypdf](https://ocrmypdf.readthedocs.io) (Tesseract) to gain a text layer, and the result is parsed
instead. A junk text layer triggers a second pass with `--force-ocr`.

- OCR runs at roughly 1–3 seconds per page, so the first ingest of a scanned module takes a while;
  bulk-load via `scripts/ingest.py` or the Airflow DAG rather than waiting on an HTTP upload.
- Results are cached in `data/.ocr-cache`, keyed by file content — re-indexing never re-runs OCR.
- The response from `/api/v1/notes` reports `"ocr": true` when a text layer had to be created.
- Tune with the `OCR_*` settings in `.env`; other languages need the matching Tesseract pack added to
  the Dockerfile (e.g. `tesseract-ocr-kan` for Kannada, then `OCR_LANGUAGE=eng+kan`).

Subject names and module titles come from [data/catalog.yaml](data/catalog.yaml), which is seeded for the
**CSE 2022 scheme**. Check it against the official VTU syllabus and extend it for your branch.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Status of Postgres, OpenSearch, Redis, LLM, embeddings, Langfuse |
| GET | `/api/v1/subjects?branch=&scheme=&semester=` | List subjects |
| GET | `/api/v1/subjects/{code}/modules` | Modules, with note and indexed-note counts |
| POST | `/api/v1/notes` | Upload and index a note |
| POST | `/api/v1/notes/sync?force=` | Index new or changed notes from `data/` |
| GET/DELETE | `/api/v1/notes`, `/api/v1/notes/{id}` | Inspect or remove notes |
| POST | `/api/v1/search` | Hybrid, BM25 or vector search over chunks |
| POST | `/api/v1/ask` | Quick answer: one retrieval, then generation |
| POST | `/api/v1/agentic-ask` | LangGraph agent with guardrail, grading and query rewriting |

Search, ask and agentic-ask all accept the filters `branch`, `scheme`, `semester`, `subject_code` and
`module_numbers`.

```bash
curl -X POST localhost:8000/api/v1/agentic-ask -H 'content-type: application/json' \
  -d '{"question": "Explain dual-mode operation", "subject_code": "BCS303"}'
```

The response includes the answer, numbered `sources` (with a `cited` flag), the guardrail verdict, any
rewritten queries, and a step-by-step `steps` log of the agent's run.

## How it works

- **Chunking** ([chunker.py](src/vtu_rag/ingestion/chunker.py)): splits notes at detected headings
  (markdown `#`, `2.3 Paging`, `MODULE 2`, short ALL-CAPS lines), then packs each section into chunks of
  about 350 words with a 60-word overlap. Chunks keep their heading path (`Module 1 > 1.3 Dual-Mode
  Operation`) and page range. Very small sections are merged into the next one.
- **Search** ([search/](src/vtu_rag/services/search/)): OpenSearch `hybrid` query (BM25 plus Lucene HNSW
  k-NN) with a min-max normalisation pipeline (weights 0.3 / 0.7). Filters apply to both halves of the
  query. Search automatically falls back to BM25 when embeddings are unavailable.
- **Agent** ([agent/graph.py](src/vtu_rag/agent/graph.py)):
  - The guardrail scores whether the question is in scope. If the classifier fails, the question is
    allowed through.
  - The grader picks the relevant excerpts. When none are relevant, the question is rewritten and
    retrieved again, up to `AGENT_MAX_REWRITES` times.
  - Generation must cite `[n]` and must say so when the notes don't contain the answer.
- **LLM** ([services/llm/](src/vtu_rag/services/llm/)): a `LLMProvider` interface with an Ollama provider
  (the default) and an OpenAI-compatible provider. To use Groq or Together, set `LLM_PROVIDER=groq`,
  `LLM_API_KEY` and `LLM_API_MODEL`.
- **Caching:** answers are cached in Redis, keyed by question, filters and model. The cache is
  invalidated whenever notes are re-indexed.
- **Tracing:** Langfuse starts with a project and API keys already created (headless init), so every
  node and LLM call is traced out of the box — sign in at http://localhost:3000 to see them. Clear
  `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` to turn tracing off; a Langfuse outage never fails a
  request.
- **Scraping:** `ScraperSource` in [sources.py](src/vtu_rag/ingestion/sources.py) is the extension point.
  It isn't implemented yet.

## Optional pieces

```bash
docker compose --profile telegram up -d telegram-bot     # needs TELEGRAM_BOT_TOKEN
docker compose --profile dashboards up -d                # OpenSearch Dashboards on :5601
```

## Development

```bash
uv sync                       # Python 3.11–3.13
uv run pytest                 # unit tests (no services needed)
uv run ruff check src tests

# Run the API on the host against the compose services
docker compose up -d postgres opensearch redis ollama
POSTGRES_HOST=localhost OPENSEARCH_HOST=http://localhost:9200 REDIS_HOST=localhost \
OLLAMA_HOST=http://localhost:11434 DATA_DIR=data uv run uvicorn vtu_rag.main:app --reload
```

Layout:

```
src/vtu_rag/
  config.py          settings (env-driven)          container.py   service wiring
  models/            Subject, Module, Note, Chunk   repositories/  DB access
  ingestion/         path parser, parsers, chunker, sources, pipeline
  services/          embeddings, search, llm, rag, cache, tracing
  agent/             LangGraph state, graph, service
  routers/           FastAPI endpoints              schemas/       request/response models
  ui/                Gradio app                     bots/          Telegram bot
airflow/dags/        re-index DAG
scripts/ingest.py    ingestion CLI
```

## Acknowledgements

The architecture was inspired by studying the open-source
[production-agentic-rag-course](https://github.com/jamwithai/production-agentic-rag-course) (MIT). This
project is an independent implementation for a different domain.

## License

MIT
