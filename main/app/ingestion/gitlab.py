"""GitLab issue fetch + normalization.

Mirrors the github.py interface so sync_repos.py can call either connector
interchangeably.  Uses the python-gitlab library (pip install python-gitlab).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from app.core.config import get_settings
from app.core.db import get_conn, upsert_issue
from app.core.schema import Issue
from app.ingestion.preprocess import run_after_upsert


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def _strip_markdown(text: str) -> str:
    """Minimal markdown → plain-text for body_plain (mirrors github.py)."""
    if not text:
        return ""
    s = re.sub(r"```[\s\S]*?```", " ", text)
    s = re.sub(r"`[^`]+`", " ", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    s = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", s)
    s = re.sub(r"^#{1,6}\s*", "", s, flags=re.MULTILINE)
    s = re.sub(r"\n+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _gl_issue_to_canonical(namespace: str, project_name: str, gl_issue) -> Issue:
    body = gl_issue.description or ""
    body_plain = _strip_markdown(body)
    labels = gl_issue.labels or []

    return Issue(
        source="gitlab",
        repo_owner=namespace,
        repo_name=project_name,
        issue_number=gl_issue.iid,
        title=gl_issue.title or "",
        body=body,
        body_plain=body_plain,
        state="open" if gl_issue.state == "opened" else "closed",
        labels=labels,
        created_at=_parse_iso(gl_issue.created_at) or datetime.now(timezone.utc),
        updated_at=_parse_iso(gl_issue.updated_at) or datetime.now(timezone.utc),
        closed_at=_parse_iso(getattr(gl_issue, "closed_at", None)),
        url=gl_issue.web_url or "",
        text_full="",  # set by run_after_upsert
    )


def sync_gitlab_project(
    namespace: str,
    project_name: str,
    token: str | None = None,
    gitlab_url: str | None = None,
) -> tuple[int, int]:
    """Fetch all issues (open + closed) for a GitLab project and upsert them.

    Args:
        namespace:    GitLab namespace (user or group slug).
        project_name: Project slug within the namespace.
        token:        Private token; falls back to GITLAB_TOKEN env var.
        gitlab_url:   GitLab instance URL; falls back to GITLAB_URL env var.

    Returns:
        (inserted_or_updated_count, skipped_count)
    """
    settings = get_settings()
    auth_token = token or settings.gitlab_token
    if not auth_token:
        raise ValueError("GITLAB_TOKEN is required for sync_gitlab_project")

    try:
        import gitlab  # noqa: PLC0415
    except ImportError:
        raise ImportError("python-gitlab is required: pip install python-gitlab")

    gl_url = gitlab_url or settings.gitlab_url or "https://gitlab.com"
    gl = gitlab.Gitlab(gl_url, private_token=auth_token)

    project_path = f"{namespace}/{project_name}"
    project = gl.projects.get(project_path)

    updated = 0
    skipped = 0
    with get_conn() as conn:
        for gl_issue in project.issues.list(state="all", iterator=True):
            if getattr(gl_issue, "merge_request_count", None) is not None and \
               gl_issue.type == "MERGE_REQUEST_REVIEW_NOTE":
                skipped += 1
                continue
            issue = _gl_issue_to_canonical(namespace, project_name, gl_issue)
            issue_id = upsert_issue(conn, issue)
            run_after_upsert(
                conn,
                issue_id,
                issue,
                repo_key=f"{namespace}/{project_name}",
                prepend_labels=True,
            )
            updated += 1
        conn.commit()

    return (updated, skipped)
