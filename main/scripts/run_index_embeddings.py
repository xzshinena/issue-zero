#!/usr/bin/env python3
"""
Compute embeddings for all issues (text_full + chunks) and upsert into issue_embeddings.
Run after sync. Requires 004_issue_embeddings.sql and sentence-transformers (or OpenAI).

Run from main/:  python scripts/run_index_embeddings.py
"""

import os
import sys

_MAIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MAIN not in sys.path:
    sys.path.insert(0, _MAIN)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_MAIN, ".env"))
    load_dotenv(os.path.join(os.path.dirname(_MAIN), ".env"))
except ImportError:
    pass

from app.ingestion.pipeline import run_index_embeddings


def main() -> int:
    try:
        issues_ok, emb_count = run_index_embeddings()
        print(f"Indexed {issues_ok} issues, {emb_count} embeddings.")
        return 0
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
