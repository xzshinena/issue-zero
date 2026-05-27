"""Tests for BM25 dict-keyed cache."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def reset_bm25_cache():
    import app.retrieval.hybrid as hybrid_mod
    hybrid_mod._bm25_cache = {}
    yield
    hybrid_mod._bm25_cache = {}


def _make_conn(rows=None):
    conn = MagicMock()
    cur = MagicMock()
    cur.__enter__ = lambda s: s
    cur.__exit__ = MagicMock(return_value=False)
    cur.fetchall.return_value = rows or []
    conn.cursor.return_value = cur
    return conn


def test_bm25_cache_unfiltered():
    from app.retrieval.hybrid import get_bm25_index

    conn = _make_conn()
    idx = get_bm25_index(conn, repo_filter=None)
    assert idx is not None


def test_bm25_cache_filtered_different_from_unfiltered():
    from app.retrieval.hybrid import get_bm25_index

    conn = _make_conn()
    unfiltered = get_bm25_index(conn, repo_filter=None)
    filtered = get_bm25_index(conn, repo_filter="owner/repo")
    assert unfiltered is not filtered


def test_bm25_cache_rebuild():
    from app.retrieval.hybrid import get_bm25_index

    conn = _make_conn()
    first = get_bm25_index(conn, repo_filter=None)
    second = get_bm25_index(conn, repo_filter=None, rebuild=True)
    assert first is not second
