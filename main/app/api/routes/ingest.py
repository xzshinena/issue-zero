"""Ingestion API routes.

Endpoints:
- POST /ingest        — trigger a GitHub repo sync + embed in the background
- POST /ingest/gitlab — trigger a GitLab project sync + embed in the background
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from app.ingestion.tasks import enqueue_sync, enqueue_sync_gitlab

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ingest"])


class IngestRequest(BaseModel):
    repo: str = Field(..., description="GitHub repo in 'owner/name' format")
    index: bool = Field(True, description="Also compute embeddings after sync")


class IngestGitLabRequest(BaseModel):
    namespace: str = Field(..., description="GitLab namespace (user or group slug)")
    project: str = Field(..., description="GitLab project slug")
    index: bool = Field(True, description="Also compute embeddings after sync")


class IngestResponse(BaseModel):
    status: str
    repo: str
    queued: bool = Field(False, description="True when dispatched to Celery")


def _sync_task(owner: str, repo_name: str, index: bool) -> None:
    try:
        from app.ingestion.github import sync_repo  # noqa: PLC0415
        from app.ingestion.pipeline import run_index_embeddings  # noqa: PLC0415

        sync_repo(owner, repo_name)
        if index:
            run_index_embeddings()
    except Exception:
        logger.exception("Sync failed for %s/%s", owner, repo_name)


def _sync_gitlab_task(namespace: str, project_name: str, index: bool) -> None:
    try:
        from app.ingestion.gitlab import sync_gitlab_project  # noqa: PLC0415
        from app.ingestion.pipeline import run_index_embeddings  # noqa: PLC0415

        sync_gitlab_project(namespace, project_name)
        if index:
            run_index_embeddings()
    except Exception:
        logger.exception("Sync failed for %s/%s", namespace, project_name)


@router.post("/ingest", response_model=IngestResponse, status_code=202)
def trigger_ingest(request: IngestRequest, background_tasks: BackgroundTasks):
    """Trigger a GitHub repo sync (and optional embedding index) in the background.

    Dispatches to Celery when configured; falls back to FastAPI BackgroundTasks.
    """
    owner, sep, repo_name = request.repo.partition("/")
    if not sep or not owner.strip() or not repo_name.strip():
        raise HTTPException(status_code=422, detail="repo must be 'owner/name'")

    owner, repo_name = owner.strip(), repo_name.strip()

    queued = enqueue_sync(owner, repo_name, index=request.index)
    if not queued:
        background_tasks.add_task(_sync_task, owner, repo_name, request.index)

    return IngestResponse(status="accepted", repo=f"{owner}/{repo_name}", queued=queued)


@router.post("/ingest/gitlab", response_model=IngestResponse, status_code=202)
def trigger_ingest_gitlab(request: IngestGitLabRequest, background_tasks: BackgroundTasks):
    """Trigger a GitLab project sync (and optional embedding index) in the background.

    Dispatches to Celery when configured; falls back to FastAPI BackgroundTasks.
    """
    namespace = request.namespace.strip()
    project = request.project.strip()
    if not namespace or not project:
        raise HTTPException(status_code=422, detail="namespace and project are required")

    queued = enqueue_sync_gitlab(namespace, project, index=request.index)
    if not queued:
        background_tasks.add_task(_sync_gitlab_task, namespace, project, request.index)

    return IngestResponse(
        status="accepted", repo=f"{namespace}/{project}", queued=queued
    )
