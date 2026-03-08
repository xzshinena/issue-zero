"""
Ingest API route: trigger GitHub issue sync for a repo.

POST /ingest — enqueue or run sync for a repo.
"""

from __future__ import annotations

import threading

from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from app.core.config import get_settings

router = APIRouter(tags=["ingest"])


# ---------------------------------------------------------------------------
# Auth (simple API-key gate)
# ---------------------------------------------------------------------------

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
VALID_API_KEYS = {"dev-key-12345", "test-key-67890"}


def require_api_key(api_key: str | None = Security(API_KEY_HEADER)):
    if api_key is None:
        raise HTTPException(status_code=401, detail="Missing API key. Pass X-API-Key header.")
    if api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API key.")
    return api_key


# ---------------------------------------------------------------------------
# Request / Response
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    repo: str = Field(..., description="owner/repo, e.g. 'rust-lang/rust'")
    full_sync: bool = Field(False, description="If true, sync all issues (open+closed)")
    index_embeddings: bool = Field(True, description="Run embedding index after sync")


class IngestResponse(BaseModel):
    message: str
    repo: str
    full_sync: bool


# ---------------------------------------------------------------------------
# Background sync helper
# ---------------------------------------------------------------------------

def _run_sync_background(owner: str, name: str, index: bool) -> None:
    """Run sync + optional embedding in a background thread."""
    from app.ingestion.github import sync_repo
    from app.ingestion.pipeline import run_index_embeddings

    try:
        sync_repo(owner, name)
        if index:
            run_index_embeddings()
    except Exception:
        import traceback
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/ingest", response_model=IngestResponse, status_code=202)
def trigger_ingest(
    request: IngestRequest,
    api_key: str = Depends(require_api_key),
):
    """
    Trigger GitHub issue sync for a given repo.
    Runs in a background thread; returns 202 Accepted immediately.
    """
    parts = request.repo.strip().split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise HTTPException(status_code=400, detail="repo must be 'owner/name'")
    owner, name = parts

    settings = get_settings()
    if not settings.github_token:
        raise HTTPException(status_code=500, detail="GITHUB_TOKEN not configured on server.")

    thread = threading.Thread(
        target=_run_sync_background,
        args=(owner, name, request.index_embeddings),
        daemon=True,
    )
    thread.start()

    return IngestResponse(
        message=f"Ingestion started for {request.repo} (background).",
        repo=request.repo,
        full_sync=request.full_sync,
    )
