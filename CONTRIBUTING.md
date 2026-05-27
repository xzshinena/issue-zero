# Contributing to Issue Zero

Thanks for your interest in contributing. This document covers everything you need to go from zero to a merged pull request.

---

## Table of contents

- [Development setup](#development-setup)
- [Project structure](#project-structure)
- [How the pieces fit together](#how-the-pieces-fit-together)
- [Running tests](#running-tests)
- [Code style](#code-style)
- [Making a change](#making-a-change)
- [Pull request checklist](#pull-request-checklist)
- [Good first issues](#good-first-issues)
- [Reporting bugs](#reporting-bugs)

---

## Development setup

### Prerequisites

| Tool | Version |
|---|---|
| Python | 3.11+ |
| Node | 18+ (frontend only) |
| PostgreSQL | 14+ with [pgvector](https://github.com/pgvector/pgvector) |
| Git | any recent version |

### 1. Fork and clone

```bash
git clone https://github.com/<your-fork>/issue-zero.git
cd issue-zero
```

### 2. Python environment

```bash
cd main
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
```

Minimum required values for local development:

```dotenv
# A local Postgres instance works fine; pgvector extension must be enabled
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/issue_zero_dev

# A fine-grained PAT with issues:read is enough for testing
GITHUB_TOKEN=github_pat_...

# Use the local model to avoid needing an OpenAI key
EMBEDDING_PROVIDER=sentence-transformers
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
EMBEDDING_DIM=384
```

### 4. Set up the database

```bash
# Create the database
createdb issue_zero_dev

# Enable pgvector (requires postgres-contrib or the pgvector package)
psql issue_zero_dev -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Run migrations
python scripts/run_migrations.py
```

### 5. Seed some data (optional but useful)

```bash
python scripts/sync_repos.py --repo python/cpython --index
# or use a smaller repo you have access to
```

### 6. Start the API

```bash
uvicorn app.main:app --reload
# http://127.0.0.1:8000/docs
```

### 7. Frontend (optional)

```bash
cd frontend
npm install
npm run dev     # http://localhost:5173 — proxies API to :8000
```

---

## Project structure

```
main/
├── app/
│   ├── api/
│   │   └── routes/
│   │       ├── search.py       # POST /search, GET /related/{id}
│   │       └── ingest.py       # POST /ingest
│   ├── core/
│   │   ├── config.py           # Pydantic settings (all env vars)
│   │   ├── db.py               # psycopg connection pool + DB helpers
│   │   └── schema.py           # Canonical Issue model
│   ├── ingestion/
│   │   ├── github.py           # GitHub API sync
│   │   ├── preprocess.py       # Text cleaning, chunking, text_full builder
│   │   ├── pipeline.py         # run_index_embeddings() orchestrator
│   │   └── tasks.py            # Celery task wrapper (optional)
│   ├── ml/
│   │   ├── classifiers.py      # predict() — SetFit → sklearn → heuristic
│   │   ├── label_schema.py     # Canonical label lists for all 4 tasks
│   │   └── train/
│   │       ├── trainer.py      # train_all(), LabeledClassifier
│   │       ├── data_loader.py  # JSONL loader + train/val split
│   │       └── feature_extractor.py  # embed texts → numpy array
│   ├── rag/
│   │   └── pack_builder.py     # Citation-only intelligence pack
│   ├── retrieval/
│   │   ├── embedder.py         # Singleton embedder (sentence-transformers or OpenAI)
│   │   ├── hybrid.py           # BM25 + pgvector + RRF fusion
│   │   └── reranker.py         # CrossEncoder reranker
│   └── main.py                 # FastAPI app, lifespan, error handlers, SPA mount
├── eval/
│   ├── classification_metrics.py   # Precision/recall/F1 per task
│   └── retrieval_metrics.py        # Recall@K, MRR, nDCG@K
├── frontend/
│   └── src/
│       ├── App.tsx             # UI — search form, results, ingest drawer
│       ├── api.ts              # Typed fetch wrappers
│       └── index.css           # All styles
├── migrations/                 # SQL files (001–005)
├── models/                     # Trained artifacts — gitignored
├── scripts/
│   ├── sync_repos.py
│   ├── run_index_embeddings.py
│   ├── train_classifiers.py
│   ├── train_setfit.py
│   ├── label_data.py
│   └── run_migrations.py
└── tests/
    ├── conftest.py             # Shared fixtures
    ├── unit/                   # Fast, no DB required
    └── integration/            # Requires TEST_DATABASE_URL
```

---

## How the pieces fit together

Understanding these three flows covers most of the codebase.

### Ingestion flow

```
sync_repos.py
  └─► github.py        fetch issues from GitHub API
  └─► preprocess.py    clean HTML/markdown, chunk long issues, build text_full
  └─► db.py            upsert_issue() — ON CONFLICT UPDATE (idempotent)

run_index_embeddings.py
  └─► db.py            get_issues_for_embedding() — single LEFT JOIN (no N+1)
  └─► embedder.py      embed_batch(texts, batch_size=64)
  └─► db.py            insert_issue_embedding() — upserts into issue_embeddings
```

### Search flow

```
POST /search (routes/search.py)
  └─► hybrid.py        BM25 keyword search (in-memory, per-repo cache)
                       + pgvector cosine search (HNSW index)
                       → Reciprocal Rank Fusion → top 20 candidates
  └─► reranker.py      CrossEncoder rescores pairs → top N
  └─► pack_builder.py  attach classifier predictions, pick suggested action
  └─► classifiers.py   SetFit (urgency) → sklearn LR → heuristic fallback
```

### Classification flow

```
classifiers.predict(text)
  └─► _try_load_setfit()      load models/urgency-setfit/ if present
  └─► _try_load_trained()     load *.joblib files if present
  └─► _predict_trained()      SetFit for urgency + LR for other 3 tasks
  └─► _heuristic_predict()    regex fallback — always returns something
```

---

## Running tests

```bash
cd main

# Full suite (unit tests only — no DB needed)
pytest tests/unit/

# With coverage
pytest --cov=app --cov=scripts tests/unit/

# Integration tests (need a real database)
TEST_DATABASE_URL=postgresql://... pytest tests/integration/

# A single file
pytest tests/unit/test_hybrid.py -v
```

The unit tests mock external dependencies (DB, GitHub, SetFit, datasets) — you do not need a running database or API keys to run them.

### Writing tests

- **Unit tests** go in `tests/unit/`. Mock at the boundary; don't hit the network or database.
- **Integration tests** go in `tests/integration/` and should be decorated with `pytest.mark.skipif` when `TEST_DATABASE_URL` is not set (see `conftest.py`).
- Fixtures shared across multiple test files belong in `tests/conftest.py`.
- For optional dependencies (`setfit`, `datasets`), inject fake modules via `monkeypatch.setitem(sys.modules, ...)` rather than skipping — see `test_setfit_classifier.py` for the pattern.

---

## Code style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
# Check
ruff check main/

# Fix auto-fixable issues
ruff check --fix main/

# Format
ruff format main/
```

The CI pipeline runs `ruff check` on every push. A failing lint check blocks merge.

### General conventions

- **Type annotations** on all function signatures.
- **No comments explaining what code does** — names should do that. Comments are for *why*: hidden constraints, subtle invariants, workarounds for specific bugs.
- **No docstrings on trivial functions**. A module-level docstring summarising the public API is fine.
- **No premature abstractions** — three similar lines is fine; a shared helper earns its place when there are four or more callsites.
- **Imports** — stdlib first, then third-party, then internal. Lazy imports (inside functions) are acceptable for optional heavy dependencies (`setfit`, `torch`, `joblib`).

### Frontend conventions

- TypeScript strict mode is on — no `any`.
- Styles live in `index.css` using CSS custom properties; no CSS-in-JS or external component libraries.
- Keep API types in `api.ts`; keep component state in `App.tsx` until the app grows enough to justify splitting.

---

## Making a change

### Small changes (typos, docs, single-file tweaks)

1. Fork the repo, create a branch: `git checkout -b fix/brief-description`
2. Make the change, run `ruff check main/` and `pytest tests/unit/`
3. Open a pull request — a one-paragraph description is enough

### Larger changes (new features, refactors, new dependencies)

1. **Open an issue first** and describe what you want to change and why. This avoids duplicated effort and lets maintainers give early feedback on the approach.
2. Wait for a `:+1:` or discussion before writing significant code.
3. Keep the PR focused — one logical change per PR.
4. Update or add tests.
5. If you add a new env var, add it to `.env.example` and the README table.

### Branching conventions

| Prefix | When to use |
|---|---|
| `feat/` | New feature |
| `fix/` | Bug fix |
| `refactor/` | Code change with no behaviour change |
| `test/` | Test-only change |
| `docs/` | Documentation only |
| `chore/` | Tooling, deps, CI |

### Commit messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add GitLab sync connector
fix: BM25 cache returns wrong corpus for filtered queries
test: add hybrid search unit tests
docs: clarify embedding_dim in README
chore: bump ruff to 0.6
```

Keep the subject line under 72 characters. If the change needs explanation, add it in the body after a blank line.

---

## Pull request checklist

Before marking your PR ready for review:

- [ ] `ruff check main/` passes with no errors
- [ ] `pytest tests/unit/` passes
- [ ] New behaviour has test coverage
- [ ] New env vars are documented in `.env.example` and `README.md`
- [ ] No new `print()` statements in library code (use `logging`)
- [ ] No `TODO` or `FIXME` left in the diff unless you've filed a follow-up issue

---

## Good first issues

Looking for somewhere to start? These areas are well-scoped and don't require deep context:

- **Chunk-level retrieval** — `hybrid.py` currently filters `chunk_id IS NULL`; extend the vector search to also query chunk embeddings and merge results. The schema and stored embeddings are already in place.
- **GitLab connector** — `app/core/schema.py` has a `source` field for `gitlab`; a `sync_gitlab_repo()` function in `ingestion/` following the same interface as `sync_repo()` would complete multi-source support.
- **Celery worker** — `ingestion/tasks.py` has the skeleton; wire it up with a Redis broker and update the ingest route to enqueue rather than use `BackgroundTasks`.
- **Eval datasets** — contribute annotated issue pairs for `eval/relevance_set.jsonl` or classification labels for `eval/classification_set.jsonl`.
- **Docker Compose with Postgres** — extend `compose.yaml` to include a Postgres + pgvector service so the full stack can be started with a single command.

---

## Reporting bugs

Open a [GitHub issue](https://github.com/xzshinena/issue-zero/issues) with:

1. What you were trying to do
2. What happened (include the full traceback if applicable)
3. What you expected to happen
4. Your environment: OS, Python version, database version, `EMBEDDING_PROVIDER`

For security vulnerabilities, please email directly rather than opening a public issue.
