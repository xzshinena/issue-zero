"""Tests for API key authentication dependency."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.auth import require_api_key


def _app_with_auth():
    from fastapi import Depends
    from app.core.auth import require_api_key

    app = FastAPI()

    @app.get("/protected")
    def protected(_key=Depends(require_api_key)):
        return {"ok": True}

    return app


class TestOpenAccess:
    def test_no_keys_configured_allows_any_request(self):
        client = TestClient(_app_with_auth())
        with patch("app.core.auth.get_settings") as mock_cfg:
            mock_cfg.return_value.api_keys = ""
            resp = client.get("/protected")
        assert resp.status_code == 200

    def test_blank_keys_only_allows_any_request(self):
        client = TestClient(_app_with_auth())
        with patch("app.core.auth.get_settings") as mock_cfg:
            mock_cfg.return_value.api_keys = "  ,  "
            resp = client.get("/protected")
        assert resp.status_code == 200


class TestKeyEnforcement:
    def test_valid_key_accepted(self):
        client = TestClient(_app_with_auth())
        with patch("app.core.auth.get_settings") as mock_cfg:
            mock_cfg.return_value.api_keys = "secret123"
            resp = client.get("/protected", headers={"X-API-Key": "secret123"})
        assert resp.status_code == 200

    def test_invalid_key_rejected(self):
        client = TestClient(_app_with_auth())
        with patch("app.core.auth.get_settings") as mock_cfg:
            mock_cfg.return_value.api_keys = "secret123"
            resp = client.get("/protected", headers={"X-API-Key": "wrong"})
        assert resp.status_code == 401

    def test_missing_key_rejected(self):
        client = TestClient(_app_with_auth())
        with patch("app.core.auth.get_settings") as mock_cfg:
            mock_cfg.return_value.api_keys = "secret123"
            resp = client.get("/protected")
        assert resp.status_code == 401

    def test_one_of_multiple_keys_accepted(self):
        client = TestClient(_app_with_auth())
        with patch("app.core.auth.get_settings") as mock_cfg:
            mock_cfg.return_value.api_keys = "key1,key2,key3"
            resp = client.get("/protected", headers={"X-API-Key": "key2"})
        assert resp.status_code == 200

    def test_keys_with_whitespace_trimmed(self):
        client = TestClient(_app_with_auth())
        with patch("app.core.auth.get_settings") as mock_cfg:
            mock_cfg.return_value.api_keys = " key1 , key2 "
            resp = client.get("/protected", headers={"X-API-Key": "key1"})
        assert resp.status_code == 200
