#!/usr/bin/env python3
"""
Batch sync: read REPOS_TO_SYNC from config, fetch from GitHub, normalize,
preprocess (text_full + chunks), and upsert into PostgreSQL.

Run from main/:  python scripts/sync_repos.py
Or sync a single repo:  python scripts/sync_repos.py --repo owner/repo
"""

import argparse
import os
import sys

# Ensure app is importable when run as script
_MAIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MAIN not in sys.path:
    sys.path.insert(0, _MAIN)

from app.core.config import get_settings
from app.ingestion.github import sync_repo, sync_repos_from_config
from app.ingestion.pipeline import run_index_embeddings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sync GitHub issues to PostgreSQL (fetch + normalize + preprocess + upsert)."
    )
    parser.add_argument(
        "--repo",
        metavar="OWNER/REPO",
        help="Sync a single repo (overrides REPOS_TO_SYNC).",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Only print errors.",
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="After sync, run embedding index (issue_embeddings).",
    )
    args = parser.parse_args()

    if args.repo:
        part = args.repo.strip()
        if "/" not in part:
            print("error: --repo must be OWNER/REPO", file=sys.stderr)
            return 1
        owner, _, repo_name = part.partition("/")
        owner, repo_name = owner.strip(), repo_name.strip()
        if not owner or not repo_name:
            print("error: --repo must be OWNER/REPO", file=sys.stderr)
            return 1
        try:
            updated, skipped = sync_repo(owner, repo_name)
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        if not args.quiet:
            print(f"{owner}/{repo_name}: {updated} issues upserted, {skipped} PRs skipped.")
        if args.index:
            issues_ok, emb_count = run_index_embeddings()
            if not args.quiet:
                print(f"Index: {issues_ok} issues, {emb_count} embeddings.")
        return 0

    settings = get_settings()
    raw = (settings.repos_to_sync or "").strip()
    if not raw:
        if not args.quiet:
            print("No REPOS_TO_SYNC configured. Set in .env or use --repo OWNER/REPO.")
        return 0

    try:
        results = sync_repos_from_config()
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if not args.quiet:
        for key, (updated, skipped) in results.items():
            print(f"{key}: {updated} issues upserted, {skipped} PRs skipped.")
        print(f"Done. {len(results)} repo(s) synced.")
    if args.index:
        issues_ok, emb_count = run_index_embeddings()
        if not args.quiet:
            print(f"Index: {issues_ok} issues, {emb_count} embeddings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
