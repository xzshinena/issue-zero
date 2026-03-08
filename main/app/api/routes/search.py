"""
Search-related API routes.

Endpoints:
  POST /search       — hybrid retrieval + rerank + intelligence pack
  GET  /related/{id} — find related issues for an existing issue by DB id
"""

from __future__ import annotations

import re
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.core.db import get_conn
from app.retrieval.hybrid import hybrid_search
from app.retrieval.reranker import rerank
from app.rag.pack_builder import build_pack, pack_to_dict

router = APIRouter(tags=["search"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str | None = Field(None, description="Free-text query (bug report, etc.)")
    issue_url: str | None = Field(None, description="GitHub issue URL to resolve as query")
    repo: str | None = Field(None, description="Filter results by repo (owner/name)")
    limit: int = Field(10, ge=1, le=50, description="Max similar issues to return")


class SimilarIssueOut(BaseModel):
    id: str
    url: str
    title: str
    score: float
    rerank_score: float
    snippet: str


class PredictionsOut(BaseModel):
    urgency: str
    urgency_confidence: float
    issue_type: str
    issue_type_confidence: float
    action_recommendation: str
    action_confidence: float
    is_regression: bool
    regression_confidence: float


class SearchResponse(BaseModel):
    query_text: str
    query_issue_id: str | None = None
    similar_issues: list[SimilarIssueOut]
    predictions: PredictionsOut
    suggested_next_action: str
    citation_issue_ids: list[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GH_ISSUE_URL_RE = re.compile(
    r"https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/issues/(?P<number>\d+)"
)


def _resolve_issue_url(url: str) -> tuple[str, str | None]:
    """
    Given a GitHub issue URL, fetch title+body from our DB (or raise).
    Returns (text_full, issue_id_str | None).
    """
    m = _GH_ISSUE_URL_RE.match(url.strip())
    if not m:
        raise HTTPException(status_code=400, detail=f"Cannot parse GitHub issue URL: {url}")
    owner, repo, number = m.group("owner"), m.group("repo"), int(m.group("number"))
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, text_full FROM issues WHERE repo_owner=%s AND repo_name=%s AND issue_number=%s",
                (owner, repo, number),
            )
            row = cur.fetchone()
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Issue {owner}/{repo}#{number} not found in database. Ingest it first.",
        )
    return (row[1] or "", str(row[0]))


def _run_pipeline(query_text: str, repo: str | None, limit: int, query_issue_id: str | None = None) -> dict:
    """Full pipeline: hybrid search -> rerank -> pack."""
    hits = hybrid_search(query_text, final_n=limit * 5, repo_filter=repo)
    ranked = rerank(query_text, hits, top_n=limit)
    pack = build_pack(query_text, ranked, query_issue_id=query_issue_id)
    return pack_to_dict(pack)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/search", response_model=SearchResponse)
def search_issues(request: SearchRequest):
    """
    Paste a bug report or provide a GitHub issue URL.
    Returns the intelligence pack: similar issues, predictions, suggested action.
    """
    if not request.query and not request.issue_url:
        raise HTTPException(status_code=400, detail="Provide either 'query' or 'issue_url'.")

    query_issue_id: str | None = None
    if request.issue_url:
        query_text, query_issue_id = _resolve_issue_url(request.issue_url)
    else:
        query_text = request.query or ""

    if not query_text.strip():
        raise HTTPException(status_code=400, detail="Resolved query text is empty.")

    result = _run_pipeline(query_text, request.repo, request.limit, query_issue_id)
    return result


@router.get("/related/{issue_id}", response_model=SearchResponse)
def related_issues(
    issue_id: str,
    repo: str | None = Query(None, description="Filter results by repo (owner/name)"),
    limit: int = Query(10, ge=1, le=50),
):
    """
    Find related issues for an existing issue (by its DB UUID).
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, text_full FROM issues WHERE id = %s::uuid", (issue_id,))
            row = cur.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Issue {issue_id} not found.")
    text_full = row[1] or ""
    if not text_full.strip():
        raise HTTPException(status_code=400, detail="Issue has no text_full for search.")
    result = _run_pipeline(text_full, repo, limit, query_issue_id=str(row[0]))
    return result
