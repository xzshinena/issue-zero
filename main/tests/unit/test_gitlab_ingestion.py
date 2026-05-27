"""Unit tests for the GitLab ingestion connector."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def fake_gitlab_mod(monkeypatch):
    fake = MagicMock()
    monkeypatch.setitem(sys.modules, "gitlab", fake)
    return fake


def _fake_gl_issue(iid: int = 1, state: str = "opened") -> MagicMock:
    issue = MagicMock()
    issue.iid = iid
    issue.title = f"Issue {iid}"
    issue.description = "This is a **bug** with `code`."
    issue.state = state
    issue.labels = ["bug", "p1"]
    issue.web_url = f"https://gitlab.com/ns/proj/-/issues/{iid}"
    issue.created_at = "2024-01-01T00:00:00Z"
    issue.updated_at = "2024-01-02T00:00:00Z"
    issue.closed_at = None
    return issue


class TestGlIssueToCanonicaL:
    def test_basic_mapping(self):
        from app.ingestion.gitlab import _gl_issue_to_canonical

        gl = _fake_gl_issue(iid=42, state="opened")
        issue = _gl_issue_to_canonical("ns", "proj", gl)

        assert issue.source == "gitlab"
        assert issue.repo_owner == "ns"
        assert issue.repo_name == "proj"
        assert issue.issue_number == 42
        assert issue.state == "open"
        assert "bug" in issue.labels

    def test_closed_state(self):
        from app.ingestion.gitlab import _gl_issue_to_canonical

        gl = _fake_gl_issue(state="closed")
        issue = _gl_issue_to_canonical("ns", "proj", gl)
        assert issue.state == "closed"

    def test_markdown_stripped_from_body_plain(self):
        from app.ingestion.gitlab import _gl_issue_to_canonical

        gl = _fake_gl_issue()
        issue = _gl_issue_to_canonical("ns", "proj", gl)
        assert "**" not in issue.body_plain
        assert "`" not in issue.body_plain


class TestSyncGitlabProject:
    def test_happy_path(self, fake_gitlab_mod, monkeypatch):
        gl_issue = _fake_gl_issue()
        project = MagicMock()
        project.issues.list.return_value = [gl_issue]

        gl_instance = MagicMock()
        gl_instance.projects.get.return_value = project
        fake_gitlab_mod.Gitlab.return_value = gl_instance

        conn = MagicMock()
        conn.__enter__ = lambda s: s
        conn.__exit__ = MagicMock(return_value=False)

        with patch("app.ingestion.gitlab.get_conn", return_value=conn), \
             patch("app.ingestion.gitlab.upsert_issue", return_value="uuid-1"), \
             patch("app.ingestion.gitlab.run_after_upsert"), \
             patch("app.ingestion.gitlab.get_settings") as mock_cfg:
            mock_cfg.return_value.gitlab_token = "tok"
            mock_cfg.return_value.gitlab_url = "https://gitlab.com"
            from app.ingestion.gitlab import sync_gitlab_project
            updated, skipped = sync_gitlab_project("ns", "proj", token="tok")

        assert updated == 1
        assert skipped == 0

    def test_raises_without_token(self, monkeypatch):
        with patch("app.ingestion.gitlab.get_settings") as mock_cfg:
            mock_cfg.return_value.gitlab_token = ""
            mock_cfg.return_value.gitlab_url = "https://gitlab.com"
            from app.ingestion.gitlab import sync_gitlab_project
            with pytest.raises(ValueError, match="GITLAB_TOKEN"):
                sync_gitlab_project("ns", "proj")

    def test_raises_without_python_gitlab(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "gitlab", None)
        with patch("app.ingestion.gitlab.get_settings") as mock_cfg:
            mock_cfg.return_value.gitlab_token = "tok"
            mock_cfg.return_value.gitlab_url = "https://gitlab.com"
            from importlib import reload
            import app.ingestion.gitlab as gl_mod
            reload(gl_mod)
            with pytest.raises((ImportError, TypeError)):
                gl_mod.sync_gitlab_project("ns", "proj", token="tok")
