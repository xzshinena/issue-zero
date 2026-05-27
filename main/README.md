# Issue Zero

Cross-repo issue intelligence search engine. Paste a bug report (or a GitHub issue URL) and Issue Zero finds the most relevant past issues across your repositories, classifies the new issue, and returns a structured "intelligence pack" to help maintainers triage faster.

## Quick start

```bash
cd main
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # fill in DATABASE_URL and GITHUB_TOKEN at minimum
uvicorn app.main:app --reload
```

API: <http://127.0.0.1:8000>  
Health: <http://127.0.0.1:8000/health>  
Swagger: <http://127.0.0.1:8000/docs>

## Frontend (optional)

The SPA is served automatically by the API when `main/frontend/dist/` exists.

```bash
cd main/frontend
npm install
npm run build        # outputs to main/frontend/dist/
```

For development with hot-reload (proxies API calls to `localhost:8000`):

```bash
npm run dev          # starts at http://localhost:5173
```

## Environment variables

Copy `.env.example` to `.env` and fill in values. All keys are loaded by Pydantic Settings from `app/core/config.py`.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DATABASE_URL` | yes¹ | `postgresql://localhost/issue_zero` | Full PostgreSQL connection string. |
| `DB_HOST` | yes¹ | — | DB hostname. When set, takes priority over `DATABASE_URL`. |
| `DB_PORT` | — | `5432` | DB port. |
| `DB_USER` | — | — | DB user (used with `DB_HOST`). |
| `DB_PASSWORD` | — | — | DB password (used with `DB_HOST`; special chars safe). |
| `DB_NAME` | — | `tsdb` | Database name (used with `DB_HOST`). |
| `DB_SSLMODE` | — | `require` | SSL mode passed to psycopg. |
| `GITHUB_TOKEN` | yes | — | PAT with `repo` scope for syncing issues. |
| `REPOS_TO_SYNC` | — | — | Comma-separated `owner/repo` list for batch sync. |
| `EMBEDDING_PROVIDER` | — | `sentence-transformers` | `sentence-transformers` (local) or `openai`. |
| `EMBEDDING_MODEL_NAME` | — | `all-MiniLM-L6-v2` | Model name for the chosen provider. |
| `EMBEDDING_DIM` | — | `384` | Vector dimension — must match the model and pgvector index. |
| `OPENAI_API_KEY` | — | — | Required when `EMBEDDING_PROVIDER=openai`. |
| `CELERY_BROKER_URL` | — | `redis://localhost:6379/0` | Only needed if using Celery workers. |

¹ Set either `DATABASE_URL` **or** the `DB_*` group. The individual parts are preferred when the password contains special characters.

## Sync issues (batch)

```bash
cd main
python scripts/sync_repos.py                   # all repos in REPOS_TO_SYNC
python scripts/sync_repos.py --repo owner/repo # single repo
```

After sync, build the embedding index:

```bash
python scripts/run_index_embeddings.py
# or in one step:
python scripts/sync_repos.py --index
```

## Train classifiers

```bash
# SetFit urgency model (requires setfit + datasets)
python scripts/train_setfit.py
python scripts/train_setfit.py --dataset owner/ds --output models/urgency-setfit --epochs 2

# sklearn classifiers (issue_type, action_recommendation, is_regression)
python scripts/train_classifiers.py
```

Trained models are saved to `main/models/` and loaded automatically at startup.

## Docker

```bash
# API only (no frontend build needed)
docker build --target api -t issue-zero-api .
docker run -p 8000:8000 --env-file main/.env issue-zero-api

# Full image with frontend (requires main/frontend/dist/ to exist)
docker build -t issue-zero .
docker run -p 8000:8000 --env-file main/.env issue-zero
```

## Tests

```bash
cd main
pytest                          # all tests
pytest tests/unit/              # unit tests only
pytest --cov=app tests/         # with coverage
```
