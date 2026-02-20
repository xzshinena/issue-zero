"""GitHub fetch + normalization logic."""

import re
from datetime import datetime, timezone
from github import Github
from github.Issue import Issue as GhIssue
from github.Repository import Repository

from app.core.config import get_settings
from app.core.db import get_conn, upsert_issue
from app.core.schema import Issue
from app.ingestion.preprocess import run_after_upsert


def _strip_html(text: str) -> str:
    """Remove HTML tags; unescape entities."""
    if not text:
        return ""
    # Remove tags
    plain = re.sub(r"<[^>]+>", " ", text)
    # Normalize whitespace
    plain = re.sub(r"\s+", " ", plain).strip()
    return plain


def _markdown_to_plain(md: str) -> str:
    """Rough strip of markdown for BM25: code blocks, links, emphasis, headers."""
    if not md:
        return ""
    s = md
    # Code blocks (fenced)
    s = re.sub(r"```[\s\S]*?```", " ", s)
    s = re.sub(r"`[^`]+`", " ", s)
    # Links: [text](url) -> text
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    # Bold/italic
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"\*([^*]+)\*", r"\1", s)
    s = re.sub(r"__([^_]+)__", r"\1", s)
    s = re.sub(r"_([^_]+)_", r"\1", s)
    # Headers: # ## ### -> text
    s = re.sub(r"^#{1,6}\s*", "", s, flags=re.MULTILINE)
    # Horizontal rules and list markers
    s = re.sub(r"^[-*]\s+", "", s, flags=re.MULTILINE)
    s = re.sub(r"^\d+\.\s+", "", s, flags=re.MULTILINE)
    s = re.sub(r"\n+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _naive_utc(dt: datetime | None) -> datetime | None:
    """Ensure datetime is timezone-aware UTC for DB (PyGithub may return naive or aware)."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _gh_issue_to_canonical(repo_owner: str, repo_name: str, gh: GhIssue) -> Issue:
    """Map a PyGithub Issue to canonical Issue schema."""
    body_raw = gh.body or ""
    body = _strip_html(body_raw) if body_raw.strip().startswith("<") else body_raw
    if not body and body_raw:
        body = body_raw
    body_plain = _markdown_to_plain(body)
    title = gh.title or ""
    # text_full set by preprocessing after upsert (clean + optional labels + chunking)

    state = "open" if gh.state == "open" else "closed"
    labels = [lb.name for lb in gh.labels]

    return Issue(
        source="github",
        repo_owner=repo_owner,
        repo_name=repo_name,
        issue_number=gh.number,
        title=title,
        body=body,
        body_plain=body_plain,
        state=state,
        labels=labels,
        created_at=_naive_utc(gh.created_at) or datetime.now(timezone.utc),
        updated_at=_naive_utc(gh.updated_at) or datetime.now(timezone.utc),
        closed_at=_naive_utc(gh.closed_at),
        url=gh.html_url or "",
        text_full="",  # set by run_after_upsert
    )


def sync_repo(owner: str, repo_name: str, token: str | None = None) -> tuple[int, int]:
    """
    List all issues (open + closed) for the given repo, map to canonical schema, and
    idempotent upsert into issues. Skips pull requests (only issues).
    Returns (inserted_or_updated_count, skipped_count e.g. PRs).
    """
    settings = get_settings()
    auth = token or settings.github_token
    if not auth:
        raise ValueError("GITHUB_TOKEN is required for sync_repo")
    g = Github(auth)
    repo: Repository = g.get_repo(f"{owner}/{repo_name}")
    updated = 0
    skipped = 0
    with get_conn() as conn:
        for gh_issue in repo.get_issues(state="all"):
            if gh_issue.pull_request is not None:
                skipped += 1
                continue
            issue = _gh_issue_to_canonical(owner, repo_name, gh_issue)
            issue_id = upsert_issue(conn, issue)
            run_after_upsert(
                conn,
                issue_id,
                issue,
                repo_key=f"{owner}/{repo_name}",
                prepend_labels=True,
            )
            updated += 1
        conn.commit()
    return (updated, skipped)


def sync_repos_from_config() -> dict[str, tuple[int, int]]:
    """
    Sync all repos in REPOS_TO_SYNC (comma-separated owner/repo).
    Returns dict mapping "owner/repo" -> (upserted_count, skipped_count).
    """
    settings = get_settings()
    raw = (settings.repos_to_sync or "").strip()
    if not raw:
        return {}
    results: dict[str, tuple[int, int]] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part or "/" not in part:
            continue
        owner, _, repo_name = part.partition("/")
        owner, repo_name = owner.strip(), repo_name.strip()
        if not owner or not repo_name:
            continue
        key = f"{owner}/{repo_name}"
        results[key] = sync_repo(owner, repo_name)
    return results
