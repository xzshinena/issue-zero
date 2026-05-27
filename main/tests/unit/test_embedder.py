"""Tests for embedder singleton and empty-text handling."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def reset_embedder():
    import app.retrieval.embedder as emb_mod
    emb_mod._EMBEDDER = None
    yield
    emb_mod._EMBEDDER = None


def test_embedder_singleton():
    from app.retrieval.embedder import _get_embedder

    with patch("app.retrieval.embedder._SentenceTransformerEmbedder") as MockEmb:
        MockEmb.return_value = MagicMock()
        e1 = _get_embedder()
        e2 = _get_embedder()
    assert e1 is e2
    assert MockEmb.call_count == 1


def test_embed_empty_text_returns_zeros():
    from app.retrieval.embedder import _SentenceTransformerEmbedder
    from app.core.config import get_settings

    settings = get_settings()
    emb = _SentenceTransformerEmbedder(settings)
    result = emb.embed("")
    assert all(v == 0.0 for v in result)
    assert len(result) == (settings.embedding_dim or 384)
