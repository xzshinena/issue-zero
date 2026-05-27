"""Tests for BM25 dict-keyed cache and chunk-level vector search."""

from __future__ import annotations

import uuid
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


# ---------------------------------------------------------------------------
# Chunk vector search
# ---------------------------------------------------------------------------

def _chunk_rows(n_issues: int = 3, chunks_per_issue: int = 2):
    """Generate fake DB rows: (issue_id, similarity, title, url, text_full)."""
    rows = []
    for i in range(n_issues):
        iid = uuid.uuid4()
        for j in range(chunks_per_issue):
            sim = 0.9 - i * 0.1 - j * 0.05
            rows.append((iid, sim, f"Issue {i}", f"https://gh/{i}", f"text {i}"))
    return rows


def test_chunk_search_deduplicates_by_issue():
    from app.retrieval.hybrid import vector_search_chunks

    rows = _chunk_rows(n_issues=3, chunks_per_issue=3)
    conn = _make_conn(rows=rows)
    results = vector_search_chunks(conn, [0.1] * 4, top_k=10)

    issue_ids = [r[0] for r in results]
    assert len(issue_ids) == len(set(issue_ids)), "duplicate issue_ids in chunk results"


def test_chunk_search_keeps_highest_score():
    from app.retrieval.hybrid import vector_search_chunks

    iid = uuid.uuid4()
    # Two chunks for the same issue with different scores
    rows = [(iid, 0.95, "T", "U", "txt"), (iid, 0.70, "T", "U", "txt")]
    conn = _make_conn(rows=rows)
    results = vector_search_chunks(conn, [0.1] * 4, top_k=10)

    assert len(results) == 1
    assert abs(results[0][1] - 0.95) < 1e-6, "should keep the highest chunk score"


def test_chunk_search_empty_corpus():
    from app.retrieval.hybrid import vector_search_chunks

    conn = _make_conn(rows=[])
    results = vector_search_chunks(conn, [0.1] * 4, top_k=10)
    assert results == []


# ---------------------------------------------------------------------------
# hybrid_search uses pre-computed embedding when provided
# ---------------------------------------------------------------------------

def test_hybrid_search_uses_provided_embedding():
    """When query_embedding is passed, embed() must not be called."""
    from app.retrieval.hybrid import hybrid_search

    pre_computed = [0.5] * 4
    conn = _make_conn(rows=[])

    with patch("app.retrieval.hybrid.embed") as mock_embed, \
         patch("app.retrieval.hybrid.get_bm25_index") as mock_bm25, \
         patch("app.retrieval.hybrid.vector_search", return_value=[]) as _vs, \
         patch("app.retrieval.hybrid.vector_search_chunks", return_value=[]) as _vsc, \
         patch("app.retrieval.hybrid.get_conn") as mock_gc:
        mock_gc.return_value.__enter__ = lambda s: conn
        mock_gc.return_value.__exit__ = MagicMock(return_value=False)
        mock_bm25.return_value.query.return_value = []

        hybrid_search("segfault", query_embedding=pre_computed, conn=conn)

    mock_embed.assert_not_called()


def test_hybrid_search_embeds_when_no_embedding_provided():
    """When no query_embedding is passed, embed() should be called exactly once."""
    from app.retrieval.hybrid import hybrid_search

    fake_vec = [0.1] * 4
    conn = _make_conn(rows=[])

    with patch("app.retrieval.hybrid.embed", return_value=fake_vec) as mock_embed, \
         patch("app.retrieval.hybrid.get_bm25_index") as mock_bm25, \
         patch("app.retrieval.hybrid.vector_search", return_value=[]) as _vs, \
         patch("app.retrieval.hybrid.vector_search_chunks", return_value=[]) as _vsc, \
         patch("app.retrieval.hybrid.get_conn") as mock_gc:
        mock_gc.return_value.__enter__ = lambda s: conn
        mock_gc.return_value.__exit__ = MagicMock(return_value=False)
        mock_bm25.return_value.query.return_value = []

        hybrid_search("segfault", conn=conn)

    mock_embed.assert_called_once_with("segfault")


def test_hybrid_search_chunk_scores_in_source_scores():
    """chunk_vector score must appear in RetrievalHit.source_scores."""
    from app.retrieval.hybrid import hybrid_search
    import uuid as _uuid

    iid = _uuid.uuid4()
    chunk_hit = (iid, 0.88, "Title", "http://gh/1", "text")

    conn = _make_conn(rows=[])
    with patch("app.retrieval.hybrid.embed", return_value=[0.1] * 4), \
         patch("app.retrieval.hybrid.get_bm25_index") as mock_bm25, \
         patch("app.retrieval.hybrid.vector_search", return_value=[]), \
         patch("app.retrieval.hybrid.vector_search_chunks", return_value=[chunk_hit]), \
         patch("app.retrieval.hybrid.get_conn") as mock_gc:
        mock_gc.return_value.__enter__ = lambda s: conn
        mock_gc.return_value.__exit__ = MagicMock(return_value=False)
        mock_bm25.return_value.query.return_value = []

        hits = hybrid_search("error", conn=conn)

    assert len(hits) == 1
    assert "chunk_vector" in hits[0].source_scores
    assert abs(hits[0].source_scores["chunk_vector"] - 0.88) < 1e-6
