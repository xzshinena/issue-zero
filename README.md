# Issue Zero

**Cross-repo issue intelligence for maintainers.**

Paste a bug report or a GitHub issue URL. Issue Zero finds the most relevant past issues across all your repositories, classifies the new issue by urgency and type, and recommends a triage action — all grounded in your own issue history, no hallucinations.

---

## How it works

```
GitHub Repos ──► sync_repos.py ──► PostgreSQL
                                      │
                               run_index_embeddings.py
                                      │
                           ┌──────────┴──────────┐
                           │  issue_embeddings    │
                           │  (pgvector / HNSW)   │
                           └──────────┬──────────┘
                                      │
POST /search ─────────────────────────┼──────────────────────────────────┐
  query or issue_url                  │                                  │
                                      ▼                                  │
                          ┌─── Hybrid Retrieval ──────┐                  │
                          │  BM25 keyword search       │                  │
                          │  + vector cosine search    │                  │
                          │  fused via RRF (k=60)      │                  │
                          └──────────┬────────────────┘                  │
                                     │  top 20 candidates                │
                                     ▼                                   │
                          CrossEncoder reranker                          │
                          (ms-marco-MiniLM-L-6-v2)                      │
                                     │  top N results                    │
                                     ▼                                   │
                          ┌─── Multi-task Classifier ─────────────────┐  │
                          │  urgency         ◄─ SetFit (fine-tuned)   │  │
                          │  issue_type      ◄─ sklearn LR            │  │
                          │  action          ◄─ sklearn LR            │  │
                          │  is_regression   ◄─ sklearn LR            │  │
                          │  (heuristic fallback when no models)      │  │
                          └──────────────────┬────────────────────────┘  │
                                             │                           │
                                             ▼                           │
                                   Intelligence Pack  ◄──────────────────┘
                              similar issues + predictions
                              + suggested action + citations
```

The full response is served to the **React SPA** or consumed directly via the JSON API.

---

## Features

### Retrieval
- **Hybrid search** — BM25 (keyword) and pgvector (semantic) run in parallel; scores are fused with Reciprocal Rank Fusion so neither signal dominates
- **CrossEncoder reranking** — a cross-encoder rescores the top candidates for precision
- **Per-repo filtering** — scope any search to a single repo with the `repo` parameter
- **HNSW index** — approximate nearest-neighbour search stays fast as the issue corpus grows
- **Automatic chunking** — long issues (>2048 chars) are split into overlapping chunks; embeddings are stored for future chunk-level retrieval

### Classification
- **Three-tier prediction**: trained SetFit model → trained sklearn LR → regex heuristics. The system always returns a prediction; trained models are optional.
- **SetFit urgency model** — fine-tuned on `all-MiniLM-L6-v2` via contrastive learning; labels: `critical_bug · high · medium · low · enhancement · question`
- **sklearn classifiers** — LogisticRegression with balanced class weights for `issue_type`, `action_recommendation`, `is_regression`
- **Heuristic fallback** — crash patterns (SIGSEGV, OOM, panic), regression keywords, feature/docs signals, GitHub labels — zero cold-start failure

### Ingestion
- **GitHub sync** — fetches all open and closed issues (PRs excluded) via PyGithub
- **Preprocessing** — strips HTML/markdown, removes boilerplate templates (Environment, Steps to Reproduce, etc.), normalises whitespace
- **Idempotent upserts** — re-running sync never duplicates data
- **Batch embedding** — embeds in batches of 64 with a process-level singleton (no per-request model reload)

### Output
- **Citation-only intelligence pack** — every link in the response comes from the retrieved set; nothing is invented
- **Confidence scores** — every prediction includes a confidence float
- **Source scores** — BM25, vector, RRF, and rerank scores are all returned for transparency and debugging

### Deployment
- **Dual embedding backends** — `sentence-transformers` (local, 384-dim, no API key) or `openai` (1536-dim, API key required)
- **Multi-stage Docker build** — Node frontend builder + Python API stage
- **SPA served by the API** — `frontend/dist/` is mounted as static files; no separate web server needed
- **CI pipeline** — ruff lint + Docker build + pytest on every push and PR

---

## Project status

### Done
- [x] Full ingestion pipeline — GitHub and GitLab sync, HTML/markdown cleaning, chunking, batch embedding
- [x] Hybrid retrieval — BM25 + pgvector issue-level + pgvector chunk-level, all fused via RRF + CrossEncoder reranker
- [x] Embedding deduplication — query embedded once per request; shared across retrieval and classification
- [x] Multi-task ML classification — SetFit (all 4 tasks), sklearn LR (4 tasks), heuristic fallback; per-task independent loading
- [x] REST API — `POST /search`, `POST /stream-search`, `GET /related/{id}`, `POST /ingest`, `POST /ingest/gitlab`, `GET /health`
- [x] Streaming search — NDJSON server-sent events with `retrieval → reranked → complete` stages
- [x] API key authentication — `X-API-Key` header, no-op when `API_KEYS` unset (open access for local dev)
- [x] Intelligence pack builder — citation-only output, suggested action, confidence scores
- [x] React + Vite SPA — search, predictions panel, similar-issue cards, ingest drawer
- [x] Multi-stage Dockerfile + Docker Compose
- [x] GitHub Actions CI — lint, build, test
- [x] Full unit test suite (95 tests) — embedder, hybrid, ingest router, trainer, SetFit classifier, auth, streaming, GitLab, model registry
- [x] Evaluation scripts — classification metrics (precision/recall/F1) and retrieval metrics (Recall@K, MRR, nDCG@K)
- [x] Eval seed datasets — `eval/relevance_set.jsonl` and `eval/classification_set.jsonl`
- [x] `generate_eval_set.py` — auto-generate eval datasets from DB issues
- [x] SetFit training for all 4 tasks with MPS/CUDA/CPU auto-detection
- [x] Versioned model registry — `models/<task>/v<N>/` with `metadata.json` per version
- [x] Celery async job queue — Redis-backed workers with autoretry; `BackgroundTasks` fallback when Celery unavailable
- [x] GitLab connector — `POST /ingest/gitlab`, `sync_repos.py --repo gl:ns/project`
- [x] Labeled-data export script for building training datasets

### Planned
- [ ] **Multi-tenancy** — per-organisation namespacing; API keys scoped to a tenant with isolated issue corpora
- [ ] **Online learning** — re-rank feedback loop; use thumbs-up/down signals to retrain classifiers incrementally

---

## Quickstart

### Requirements
- Python 3.11+
- PostgreSQL 14+ with the [pgvector extension](https://github.com/pgvector/pgvector)
- Node 18+ (optional, for the frontend)

### 1. Clone and install

```bash
git clone https://github.com/xzshinena/issue-zero.git
cd issue-zero/main

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — set DATABASE_URL (or DB_* parts) and GITHUB_TOKEN at minimum
```

See the [full environment variable reference](#environment-variables) below.

### 3. Run migrations

```bash
python scripts/run_migrations.py --include-vector
```

### 4. Sync issues

```bash
# Add repos to REPOS_TO_SYNC in .env, then:
python scripts/sync_repos.py

# Or sync a single repo and build the embedding index in one step:
python scripts/sync_repos.py --repo owner/repo --index
```

### 5. Start the API

```bash
uvicorn app.main:app --reload
# API:     http://127.0.0.1:8000
# Swagger: http://127.0.0.1:8000/docs
# Health:  http://127.0.0.1:8000/health
```

### 6. Build and serve the frontend (optional)

```bash
cd frontend
npm install
npm run build     # outputs to frontend/dist/ — served automatically by the API
```

For frontend hot-reload during development (proxies API to `:8000`):

```bash
npm run dev       # http://localhost:5173
```

---

## Environment variables

Copy `main/.env.example` to `main/.env`.

| Variable | Required | Default | Description |
|---|---|---|---|
| `DATABASE_URL` | yes¹ | `postgresql://localhost/issue_zero` | Full PostgreSQL connection string. |
| `DB_HOST` | yes¹ | — | Hostname. When set, takes priority over `DATABASE_URL`. |
| `DB_PORT` | — | `5432` | Port. |
| `DB_USER` | — | — | User (used with `DB_HOST`). |
| `DB_PASSWORD` | — | — | Password. Special characters are URL-encoded automatically. |
| `DB_NAME` | — | `tsdb` | Database name. |
| `DB_SSLMODE` | — | `require` | SSL mode for psycopg. |
| `GITHUB_TOKEN` | yes | — | Personal access token with `repo` scope. |
| `REPOS_TO_SYNC` | — | — | Comma-separated `owner/repo` list for batch sync. |
| `EMBEDDING_PROVIDER` | — | `sentence-transformers` | `sentence-transformers` (local) or `openai`. |
| `EMBEDDING_MODEL_NAME` | — | `all-MiniLM-L6-v2` | Model name for the chosen provider. |
| `EMBEDDING_DIM` | — | `384` | Vector dimension — must match the model and the pgvector index. |
| `OPENAI_API_KEY` | — | — | Required when `EMBEDDING_PROVIDER=openai`. |
| `CELERY_BROKER_URL` | — | — | Redis URL for Celery workers. When unset, syncs run in-process. |
| `GITLAB_TOKEN` | — | — | GitLab personal access token (required for GitLab sync). |
| `GITLAB_URL` | — | `https://gitlab.com` | GitLab instance URL (for self-hosted). |
| `API_KEYS` | — | — | Comma-separated API keys. When set, all routes require `X-API-Key`. |

¹ Set either `DATABASE_URL` **or** the `DB_*` group. The individual parts are preferred when the password contains special characters.

---

## API reference

### `POST /search`

Find similar issues and get triage predictions.

```jsonc
// Request
{
  "query": "null pointer exception when opening settings on Windows",
  // or: "issue_url": "https://github.com/owner/repo/issues/42",
  "repo": "owner/repo",   // optional — filter results to one repo
  "limit": 10             // optional — 1–50, default 10
}

// Response
{
  "query_text": "...",
  "similar_issues": [
    {
      "id": "uuid",
      "url": "https://github.com/...",
      "title": "...",
      "score": 0.812,
      "rerank_score": 0.941,
      "snippet": "..."
    }
  ],
  "predictions": {
    "urgency": "high",           "urgency_confidence": 0.87,
    "issue_type": "bug",         "issue_type_confidence": 0.79,
    "action_recommendation": "assign_to_area", "action_confidence": 0.65,
    "is_regression": false,      "regression_confidence": 0.71
  },
  "suggested_next_action": "assign_to_area",
  "citation_issue_ids": ["uuid1", "uuid2"]
}
```

### `GET /related/{issue_id}`

Find related issues for an issue already in the database.

```
GET /related/550e8400-e29b-41d4-a716-446655440000?repo=owner/repo&limit=10
```

### `POST /stream-search`

Same as `/search` but streams NDJSON progress events. Useful for showing progress in the UI as each pipeline stage completes.

Three events in order:
1. `{"stage": "retrieval", "count": N, "issue_ids": [...]}`
2. `{"stage": "reranked", "count": N, "results": [...]}`
3. `{"stage": "complete", ...full SearchResponse fields...}`

Request body is identical to `/search`.

### `POST /ingest`

Trigger a GitHub repo sync (and optionally build the embedding index) in the background. Dispatches to Celery when `CELERY_BROKER_URL` is set; falls back to FastAPI `BackgroundTasks`. Returns `202 Accepted` immediately.

```jsonc
{ "repo": "owner/repo", "index": true }
// Response: {"status": "accepted", "repo": "owner/repo", "queued": true}
```

### `POST /ingest/gitlab`

Same as `/ingest` but for GitLab projects.

```jsonc
{ "namespace": "my-group", "project": "my-project", "index": true }
```

### Authentication

When `API_KEYS` is set in the environment, every `/search`, `/stream-search`, `/related`, and `/ingest` request requires an `X-API-Key` header matching one of the configured keys. When unset, all routes are open (suitable for local dev).

---

## Training classifiers

### SetFit (all four tasks)

```bash
cd main

# Train one task
python scripts/train_setfit.py \
  --task urgency \
  --dataset shinena-xiang/dev-issue-urgency-classification-ds \
  --output models/urgency-setfit \
  --epochs 1

# Train all four tasks
python scripts/train_setfit.py --task all --dataset <hf-dataset> --output models/
```

Auto-detects Apple Silicon (MPS), CUDA, or CPU.

### sklearn classifiers

```bash
# 1. Export labeled issues from your database
python scripts/label_data.py --input issues.json --output labeled.jsonl

# 2. Train all four tasks (saved to versioned models/<task>/v<N>/)
python scripts/train_classifiers.py --data labeled.jsonl --models-dir models/
```

Trained models are versioned in `main/models/<task>/v<N>/` with `metadata.json` and loaded automatically at startup. Pass `--use-registry false` to use the legacy flat layout.

### Generate eval datasets

```bash
# Auto-generate eval sets from issues already in your DB
python scripts/generate_eval_set.py --limit 500 --seed 42
```

Outputs `eval/relevance_set.jsonl` and `eval/classification_set.jsonl`.

---

## Evaluation

```bash
cd main

# Classification — per-class precision/recall/F1 and macro averages
python eval/classification_metrics.py --eval-file eval/classification_set.jsonl

# Retrieval — Recall@K, MRR, nDCG@K
python eval/retrieval_metrics.py --eval-file eval/relevance_set.jsonl --k 5 10 20
```

---

## Docker

```bash
# API only — no frontend build needed
docker build --target api -t issue-zero-api .
docker run -p 8000:8000 --env-file main/.env issue-zero-api

# Full image (API + pre-built frontend)
# Build the frontend first: cd main/frontend && npm install && npm run build
docker build -t issue-zero .
docker run -p 8000:8000 --env-file main/.env issue-zero

# Or with Compose
docker compose up
```

---

## Repository layout

```
issue-zero/
├── main/
│   ├── app/
│   │   ├── api/routes/        # FastAPI routers (search, ingest)
│   │   ├── core/              # DB pool, config, Pydantic schema
│   │   ├── ingestion/         # GitHub sync, preprocessing, embedding pipeline
│   │   ├── ml/                # Classifiers, label schema, training pipeline
│   │   ├── rag/               # Intelligence pack builder
│   │   ├── retrieval/         # Embedder, hybrid search, CrossEncoder reranker
│   │   └── main.py            # FastAPI app + lifespan
│   ├── eval/                  # Classification and retrieval evaluation scripts
│   ├── frontend/              # React + Vite SPA
│   │   └── src/               # App.tsx, api.ts, index.css
│   ├── migrations/            # SQL schema (issues, chunks, embeddings, pgvector)
│   ├── models/                # Trained model artifacts (gitignored)
│   ├── scripts/               # CLI tools (sync, train, embed, migrate, label)
│   └── tests/
│       ├── unit/              # Per-module unit tests
│       └── integration/       # E2E tests (require TEST_DATABASE_URL)
├── Dockerfile
├── compose.yaml
└── .github/workflows/ci.yml
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

MIT
