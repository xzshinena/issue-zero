"""Convert raw issue texts to embedding feature matrix."""

from __future__ import annotations

import numpy as np

from app.retrieval.embedder import embed_batch

BATCH_SIZE = 64


def extract_features(texts: list[str]) -> np.ndarray:
    """Embed a list of texts; return float32 array of shape (n, embedding_dim)."""
    if not texts:
        return np.empty((0,), dtype=np.float32)
    vecs: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        vecs.extend(embed_batch(texts[i : i + BATCH_SIZE]))
    return np.array(vecs, dtype=np.float32)
