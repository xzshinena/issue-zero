"""Unit tests for POST /stream-search (NDJSON streaming endpoint)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

_FAKE_HIT = {
    "id": "00000000-0000-0000-0000-000000000001",
    "title": "Some bug",
    "url": "https://github.com/o/r/issues/1",
    "score": 0.9,
    "rerank_score": 0.85,
    "snippet": "Some bug description",
    "source_scores": {},
}

_FAKE_PACK = {
    "query_text": "crash on startup",
    "query_issue_id": None,
    "similar_issues": [_FAKE_HIT],
    "predictions": {
        "urgency": "high",
        "urgency_confidence": 0.9,
        "issue_type": "bug",
        "issue_type_confidence": 0.8,
        "action_recommendation": "fix",
        "action_confidence": 0.7,
        "is_regression": False,
        "regression_confidence": 0.6,
    },
    "suggested_next_action": "fix immediately",
    "citation_issue_ids": ["00000000-0000-0000-0000-000000000001"],
}


def _mock_pipeline():
    return (
        patch("app.api.routes.search.embed", return_value=[0.1] * 384),
        patch("app.api.routes.search.hybrid_search", return_value=[_FAKE_HIT]),
        patch("app.api.routes.search.rerank", return_value=[_FAKE_HIT]),
        patch("app.api.routes.search.build_pack", return_value=MagicMock()),
        patch("app.api.routes.search.pack_to_dict", return_value=_FAKE_PACK),
    )


def test_stream_search_returns_three_stages():
    patches = _mock_pipeline()
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        resp = client.post("/stream-search", json={"query": "crash on startup"})

    assert resp.status_code == 200
    assert "ndjson" in resp.headers["content-type"]

    lines = [ln for ln in resp.text.strip().split("\n") if ln]
    assert len(lines) == 3

    stages = [json.loads(ln)["stage"] for ln in lines]
    assert stages == ["retrieval", "reranked", "complete"]


def test_stream_search_retrieval_stage():
    patches = _mock_pipeline()
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        resp = client.post("/stream-search", json={"query": "crash on startup"})

    lines = resp.text.strip().split("\n")
    retrieval = json.loads(lines[0])
    assert retrieval["stage"] == "retrieval"
    assert retrieval["count"] == 1
    assert isinstance(retrieval["issue_ids"], list)


def test_stream_search_reranked_stage():
    patches = _mock_pipeline()
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        resp = client.post("/stream-search", json={"query": "crash on startup"})

    lines = resp.text.strip().split("\n")
    reranked = json.loads(lines[1])
    assert reranked["stage"] == "reranked"
    assert reranked["count"] == 1
    assert reranked["results"][0]["title"] == "Some bug"


def test_stream_search_complete_stage():
    patches = _mock_pipeline()
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        resp = client.post("/stream-search", json={"query": "crash on startup"})

    lines = resp.text.strip().split("\n")
    complete = json.loads(lines[2])
    assert complete["stage"] == "complete"
    assert complete["query_text"] == "crash on startup"
    assert "predictions" in complete


def test_stream_search_missing_query():
    resp = client.post("/stream-search", json={})
    assert resp.status_code == 400


def test_stream_search_empty_query():
    resp = client.post("/stream-search", json={"query": "   "})
    assert resp.status_code == 400
