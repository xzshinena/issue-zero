"""Cross-encoder reranking: score (query, document) pairs and re-sort candidates."""

from __future__ import annotations

from dataclasses import dataclass

from app.retrieval.hybrid import RetrievalHit

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_TOP_N = 15
SNIPPET_CHARS = 400


@dataclass
class RankedHit:
    """A reranked retrieval hit with cross-encoder score and snippet."""
    issue_id: str
    title: str
    url: str
    text_full: str
    rrf_score: float
    rerank_score: float
    snippet: str
    source_scores: dict


# ---------------------------------------------------------------------------
# Cross-encoder wrapper (lazy load)
# ---------------------------------------------------------------------------

_cross_encoder = None


def _get_cross_encoder(model_name: str = DEFAULT_RERANK_MODEL):
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(model_name)
    return _cross_encoder


def _extract_snippet(text: str, max_chars: int = SNIPPET_CHARS) -> str:
    """Return the first `max_chars` characters as a snippet."""
    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_space = cut.rfind(" ")
    if last_space > max_chars // 2:
        cut = cut[:last_space]
    return cut + "..."


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rerank(
    query_text: str,
    hits: list[RetrievalHit],
    top_n: int = DEFAULT_TOP_N,
    model_name: str = DEFAULT_RERANK_MODEL,
) -> list[RankedHit]:
    """
    Rerank retrieval hits using a cross-encoder.
    Returns top_n RankedHit sorted by rerank_score descending.
    """
    if not hits:
        return []

    ce = _get_cross_encoder(model_name)
    pairs = [(query_text, h.text_full) for h in hits]
    scores = ce.predict(pairs)

    scored = list(zip(hits, scores))
    scored.sort(key=lambda pair: pair[1], reverse=True)

    results = []
    for hit, score in scored[:top_n]:
        results.append(RankedHit(
            issue_id=str(hit.issue_id),
            title=hit.title,
            url=hit.url,
            text_full=hit.text_full,
            rrf_score=hit.score,
            rerank_score=float(score),
            snippet=_extract_snippet(hit.text_full),
            source_scores=hit.source_scores,
        ))
    return results
