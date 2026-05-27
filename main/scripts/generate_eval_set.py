#!/usr/bin/env python3
"""
Generate eval datasets from the issues already in PostgreSQL.

Outputs:
  eval/relevance_set.jsonl    — query → relevant issue IDs (for retrieval eval)
  eval/classification_set.jsonl — issue text + ground-truth labels (for classifier eval)

Usage (from main/):
  python scripts/generate_eval_set.py
  python scripts/generate_eval_set.py --limit 200 --seed 42
  python scripts/generate_eval_set.py --classification-only
  python scripts/generate_eval_set.py --relevance-only
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

_MAIN = Path(__file__).resolve().parent.parent
if str(_MAIN) not in sys.path:
    sys.path.insert(0, str(_MAIN))

_EVAL_DIR = _MAIN / "eval"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_issues(conn, limit: int) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, title, body_plain, labels, urgency, issue_type,
                   action_recommendation, is_regression, repo_owner, repo_name
            FROM   issues
            WHERE  text_full IS NOT NULL AND text_full <> ''
            ORDER  BY random()
            LIMIT  %s
            """,
            (limit,),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def _issue_text(issue: dict) -> str:
    parts = [issue.get("title") or "", issue.get("body_plain") or ""]
    return " ".join(p for p in parts if p).strip()


# ---------------------------------------------------------------------------
# Classification eval
# ---------------------------------------------------------------------------

def generate_classification_set(issues: list[dict], out_path: Path) -> int:
    """Write issues that already have at least one ground-truth label column set."""
    rows = []
    for iss in issues:
        urgency = iss.get("urgency") or ""
        issue_type = iss.get("issue_type") or ""
        if not urgency and not issue_type:
            continue
        text = _issue_text(iss)
        if not text:
            continue
        rows.append({
            "id": str(iss["id"]),
            "text": text,
            "labels": iss.get("labels") or [],
            "urgency": urgency or None,
            "issue_type": issue_type or None,
            "action_recommendation": iss.get("action_recommendation") or None,
            "is_regression": iss.get("is_regression"),
            "repo": f"{iss['repo_owner']}/{iss['repo_name']}",
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return len(rows)


# ---------------------------------------------------------------------------
# Relevance eval
# ---------------------------------------------------------------------------

_QUERY_TEMPLATES = [
    ("crash {kw} in {ctx}", "crash"),
    ("null pointer exception {kw}", "crash"),
    ("memory leak when {kw}", "memory"),
    ("regression: {kw} stopped working", "regression"),
    ("feature request: add {kw}", "feature"),
    ("documentation missing for {kw}", "docs"),
    ("{kw} fails with 500 error", "error"),
    ("slow performance when {kw}", "performance"),
    ("cannot reproduce: {kw}", "flaky"),
    ("stack overflow {kw}", "crash"),
]


def _kw_from_title(title: str) -> str:
    words = title.lower().split()
    stop = {"the", "a", "an", "is", "in", "on", "at", "to", "of", "and", "or", "with", "for"}
    meaningful = [w for w in words if w not in stop and len(w) > 3]
    return " ".join(meaningful[:3]) if meaningful else title[:40]


def generate_relevance_set(issues: list[dict], out_path: Path, n_queries: int = 20) -> int:
    """Generate synthetic queries by paraphrasing issue titles."""
    rng = random.Random()
    rows = []
    sample = issues[:n_queries] if len(issues) >= n_queries else issues
    for iss in sample:
        title = (iss.get("title") or "").strip()
        if not title:
            continue
        kw = _kw_from_title(title)
        ctx = f"{iss.get('repo_owner', '')}/{iss.get('repo_name', '')}".strip("/")
        template, _ = rng.choice(_QUERY_TEMPLATES)
        query = template.format(kw=kw, ctx=ctx or "the application")
        rows.append({
            "query": query,
            "relevant_issue_ids": [str(iss["id"])],
            "source_title": title,
            "repo": f"{iss['repo_owner']}/{iss['repo_name']}",
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return len(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Generate eval datasets from DB issues.")
    parser.add_argument("--limit", type=int, default=500,
                        help="Max issues to sample from the DB (default: 500)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible sampling")
    parser.add_argument("--classification-only", action="store_true")
    parser.add_argument("--relevance-only", action="store_true")
    parser.add_argument("--out-dir", default=str(_EVAL_DIR),
                        help="Output directory for eval files")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    out_dir = Path(args.out_dir)

    from app.core.db import get_conn  # noqa: PLC0415
    try:
        with get_conn() as conn:
            print(f"Loading up to {args.limit} issues from DB …")
            issues = _load_issues(conn, args.limit)
    except Exception as exc:
        print(f"error: could not connect to DB: {exc}", file=sys.stderr)
        return 1

    if not issues:
        print("No issues found. Run sync_repos.py first to populate the DB.")
        return 0

    print(f"Loaded {len(issues)} issues.")

    if not args.relevance_only:
        n = generate_classification_set(issues, out_dir / "classification_set.jsonl")
        print(f"classification_set.jsonl: {n} labeled records written.")

    if not args.classification_only:
        n = generate_relevance_set(issues, out_dir / "relevance_set.jsonl")
        print(f"relevance_set.jsonl: {n} query records written.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
