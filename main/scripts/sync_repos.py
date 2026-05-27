#!/usr/bin/env python3
"""
Batch sync: read REPOS_TO_SYNC from config, fetch from GitHub or GitLab,
normalize, preprocess (text_full + chunks), and upsert into PostgreSQL.

Run from main/:
  python scripts/sync_repos.py                      # all repos in REPOS_TO_SYNC
  python scripts/sync_repos.py --repo owner/repo    # GitHub
  python scripts/sync_repos.py --repo gl:ns/project # GitLab (prefix with gl:)
"""

import argparse
import os
import sys

_MAIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MAIN not in sys.path:
    sys.path.insert(0, _MAIN)

from app.core.config import get_settings
from app.ingestion.github import sync_repo as sync_github, sync_repos_from_config
from app.ingestion.pipeline import run_index_embeddings


def _sync_one(repo_spec: str) -> tuple[int, int]:
    """Dispatch to GitHub or GitLab based on the repo_spec prefix."""
    if repo_spec.startswith("gl:"):
        from app.ingestion.gitlab import sync_gitlab_project  # noqa: PLC0415
        path = repo_spec[3:].strip()
        ns, _, proj = path.partition("/")
        if not ns or not proj:
            raise ValueError(f"GitLab repo must be 'gl:namespace/project', got: {repo_spec!r}")
        return sync_gitlab_project(ns.strip(), proj.strip())
    # Default: GitHub
    owner, _, repo_name = repo_spec.partition("/")
    owner, repo_name = owner.strip(), repo_name.strip()
    if not owner or not repo_name:
        raise ValueError(f"GitHub repo must be 'owner/repo', got: {repo_spec!r}")
    return sync_github(owner, repo_name)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sync issues from GitHub or GitLab to PostgreSQL."
    )
    parser.add_argument(
        "--repo",
        metavar="[gl:]OWNER/REPO",
        help="Sync a single repo. Prefix with 'gl:' for GitLab.",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Only print errors.")
    parser.add_argument("--index", action="store_true",
                        help="After sync, run embedding index (issue_embeddings).")
    args = parser.parse_args()

    if args.repo:
        spec = args.repo.strip()
        try:
            updated, skipped = _sync_one(spec)
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        if not args.quiet:
            print(f"{spec}: {updated} issues upserted, {skipped} skipped.")
        if args.index:
            issues_ok, emb_count = run_index_embeddings()
            if not args.quiet:
                print(f"Index: {issues_ok} issues, {emb_count} embeddings.")
        return 0

    # Batch from config
    settings = get_settings()
    raw = (settings.repos_to_sync or "").strip()
    if not raw:
        if not args.quiet:
            print("No REPOS_TO_SYNC configured. Set in .env or use --repo.")
        return 0

    try:
        results = sync_repos_from_config()
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if not args.quiet:
        for key, (updated, skipped) in results.items():
            print(f"{key}: {updated} issues upserted, {skipped} skipped.")
        print(f"Done. {len(results)} repo(s) synced.")

    if args.index:
        issues_ok, emb_count = run_index_embeddings()
        if not args.quiet:
            print(f"Index: {issues_ok} issues, {emb_count} embeddings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
