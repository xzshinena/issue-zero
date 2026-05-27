"""Ingestion API routes.

Endpoints:
- POST /ingest  — trigger a GitHub repo sync + embed in the background
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from app.ingestion.github import sync_repo
from app.ingestion.pipeline import run_index_embeddings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ingest"])


class IngestRequest(BaseModel):
    repo: str = Field(..., description="GitHub repo in 'owner/name' format")
    index: bool = Field(True, description="Also compute embeddings after sync")


class IngestResponse(BaseModel):
    status: str
    repo: str


def _sync_task(owner: str, repo_name: str, index: bool) -> None:
    try:
        sync_repo(owner, repo_name)
        if index:
            run_index_embeddings()
    except Exception:
        logger.exception("Sync failed for %s/%s", owner, repo_name)


@router.post("/ingest", response_model=IngestResponse, status_code=202)
def trigger_ingest(request: IngestRequest, background_tasks: BackgroundTasks):
    """Trigger a GitHub repo sync (and optional embedding index) in the background."""
    owner, sep, repo_name = request.repo.partition("/")
    if not sep or not owner.strip() or not repo_name.strip():
        raise HTTPException(status_code=422, detail="repo must be 'owner/name'")

    owner, repo_name = owner.strip(), repo_name.strip()
    background_tasks.add_task(_sync_task, owner, repo_name, request.index)
    return IngestResponse(status="accepted", repo=f"{owner}/{repo_name}")
