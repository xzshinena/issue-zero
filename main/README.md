# Issue Zero

## Environment variables

Copy `.env.example` to `.env` and fill in values. All keys are loaded via Pydantic Settings in `app/core/config.py`.

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL connection string (e.g. `postgresql://user:password@host:5432/dbname`). Used for TimescaleDB/Postgres. |
| `GITHUB_TOKEN` | GitHub personal access token (repo scope) for syncing issues/PRs. |
| `REPOS_TO_SYNC` | Comma-separated list of `owner/repo` to sync (e.g. `myorg/myapp,other/repo`). |

## Run

```bash
cd main
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Health check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
