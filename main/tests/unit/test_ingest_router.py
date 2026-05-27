"""Tests for POST /ingest."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_trigger_ingest_valid():
    with patch("app.api.routes.ingest._sync_task"):
        resp = client.post("/ingest", json={"repo": "owner/name"})
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "accepted"
    assert body["repo"] == "owner/name"


def test_trigger_ingest_missing_slash():
    resp = client.post("/ingest", json={"repo": "ownername"})
    assert resp.status_code == 422


def test_trigger_ingest_empty_owner():
    resp = client.post("/ingest", json={"repo": "/name"})
    assert resp.status_code == 422


def test_trigger_ingest_empty_repo_name():
    resp = client.post("/ingest", json={"repo": "owner/"})
    assert resp.status_code == 422


def _mock_modules(sync_side_effect=None, sync_return=(5, 0)):
    from unittest.mock import MagicMock
    import sys

    mock_gh = MagicMock()
    mock_gh.sync_repo.side_effect = sync_side_effect
    if sync_side_effect is None:
        mock_gh.sync_repo.return_value = sync_return
    mock_pipeline = MagicMock()
    mock_pipeline.run_index_embeddings.return_value = (5, 5)

    extra = {
        "github": MagicMock(),
        "app.ingestion.github": mock_gh,
        "app.ingestion.pipeline": mock_pipeline,
    }
    return patch.dict("sys.modules", extra), mock_gh


def test_sync_task_success():
    from app.api.routes.ingest import _sync_task

    ctx, mock_gh = _mock_modules()
    with ctx:
        _sync_task("owner", "repo", index=True)
    mock_gh.sync_repo.assert_called_once_with("owner", "repo")


def test_sync_task_logs_exception():
    from app.api.routes.ingest import _sync_task

    ctx, _ = _mock_modules(sync_side_effect=RuntimeError("boom"))
    with ctx, patch("app.api.routes.ingest.logger") as mock_logger:
        _sync_task("owner", "repo", index=True)
    mock_logger.exception.assert_called_once()
