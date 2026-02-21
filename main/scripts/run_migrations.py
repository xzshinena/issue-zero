#!/usr/bin/env python3
"""
Run SQL migrations using the app's DATABASE_URL (no psql required).

Run from main/:  python scripts/run_migrations.py
Runs 001_create_issues.sql and 003_issue_chunks.sql by default.
Use --include-vector to also run 002_pgvector.sql (requires vector extension).
"""

import argparse
import os
import sys

_MAIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MAIN not in sys.path:
    sys.path.insert(0, _MAIN)

# Load .env from main/ or project root so DATABASE_URL is set
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_MAIN, ".env"))
    load_dotenv(os.path.join(os.path.dirname(_MAIN), ".env"))
except ImportError:
    pass

import psycopg
from app.core.config import get_settings


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DB migrations (uses DATABASE_URL from .env).")
    parser.add_argument(
        "--include-vector",
        action="store_true",
        help="Also run 002_pgvector.sql (requires vector extension).",
    )
    args = parser.parse_args()

    settings = get_settings()
    migrations_dir = os.path.join(_MAIN, "migrations")
    files = ["001_create_issues.sql", "003_issue_chunks.sql", "004_issue_embeddings.sql"]
    if args.include_vector:
        files.insert(1, "002_pgvector.sql")

    try:
        with psycopg.connect(settings.effective_database_url) as conn:
            with conn.cursor() as cur:
                for name in files:
                    path = os.path.join(migrations_dir, name)
                    if not os.path.exists(path):
                        print(f"skip (not found): {name}", file=sys.stderr)
                        continue
                    with open(path) as f:
                        sql = f.read()
                    # Strip comments and run each statement (psql-style)
                    lines = []
                    for line in sql.splitlines():
                        s = line.strip()
                        if not s or s.startswith("--"):
                            continue
                        lines.append(line)
                    block = "\n".join(lines)
                    for stmt in block.split(";"):
                        stmt = stmt.strip()
                        if not stmt:
                            continue
                        cur.execute(stmt + ";")
                    conn.commit()
                    print(f"ok: {name}")
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
