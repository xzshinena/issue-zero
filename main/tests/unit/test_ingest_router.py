"""Tests for POST /ingest and POST /ingest/gitlab."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# /ingest (GitHub)
# ---------------------------------------------------------------------------

def test_trigger_ingest_valid_bg():
    """Falls back to BackgroundTasks when Celery unavailable."""
    with patch("app.api.routes.ingest.enqueue_sync", return_value=False), \
         patch("app.api.routes.ingest._sync_task"):
        resp = client.post("/ingest", json={"repo": "owner/name"})
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "accepted"
    assert body["repo"] == "owner/name"
    assert body["queued"] is False


def test_trigger_ingest_valid_celery():
    """Returns queued=True when Celery accepted the task."""
    with patch("app.api.routes.ingest.enqueue_sync", return_value=True):
        resp = client.post("/ingest", json={"repo": "owner/name"})
    assert resp.status_code == 202
    assert resp.json()["queued"] is True


def test_trigger_ingest_missing_slash():
    resp = client.post("/ingest", json={"repo": "ownername"})
    assert resp.status_code == 422


def test_trigger_ingest_empty_owner():
    resp = client.post("/ingest", json={"repo": "/name"})
    assert resp.status_code == 422


def test_trigger_ingest_empty_repo_name():
    resp = client.post("/ingest", json={"repo": "owner/"})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /ingest/gitlab
# ---------------------------------------------------------------------------

def test_trigger_ingest_gitlab_valid_bg():
    with patch("app.api.routes.ingest.enqueue_sync_gitlab", return_value=False), \
         patch("app.api.routes.ingest._sync_gitlab_task"):
        resp = client.post("/ingest/gitlab", json={"namespace": "ns", "project": "proj"})
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "accepted"
    assert body["repo"] == "ns/proj"
    assert body["queued"] is False


def test_trigger_ingest_gitlab_valid_celery():
    with patch("app.api.routes.ingest.enqueue_sync_gitlab", return_value=True):
        resp = client.post("/ingest/gitlab", json={"namespace": "ns", "project": "proj"})
    assert resp.status_code == 202
    assert resp.json()["queued"] is True


def test_trigger_ingest_gitlab_empty_namespace():
    resp = client.post("/ingest/gitlab", json={"namespace": "", "project": "proj"})
    assert resp.status_code == 422


def test_trigger_ingest_gitlab_empty_project():
    resp = client.post("/ingest/gitlab", json={"namespace": "ns", "project": ""})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Background task helpers
# ---------------------------------------------------------------------------

def _mock_modules(sync_side_effect=None, sync_return=(5, 0)):
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


def test_sync_gitlab_task_success():
    from app.api.routes.ingest import _sync_gitlab_task

    mock_gl = MagicMock()
    mock_pipeline = MagicMock()
    mock_pipeline.run_index_embeddings.return_value = (3, 3)

    with patch.dict("sys.modules", {
        "app.ingestion.gitlab": mock_gl,
        "app.ingestion.pipeline": mock_pipeline,
    }):
        _sync_gitlab_task("ns", "proj", index=True)

    mock_gl.sync_gitlab_project.assert_called_once_with("ns", "proj")


def test_sync_gitlab_task_logs_exception():
    from app.api.routes.ingest import _sync_gitlab_task

    mock_gl = MagicMock()
    mock_gl.sync_gitlab_project.side_effect = RuntimeError("boom")

    with patch.dict("sys.modules", {"app.ingestion.gitlab": mock_gl}), \
         patch("app.api.routes.ingest.logger") as mock_logger:
        _sync_gitlab_task("ns", "proj", index=False)

    mock_logger.exception.assert_called_once()
