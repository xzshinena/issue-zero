"""Hybrid retrieval (BM25 + vector) with Reciprocal Rank Fusion (RRF).

Search strategy:
  1. BM25 keyword search over issues.text_full (in-memory, per-repo cached)
  2. Vector cosine search on issue-level embeddings (chunk_id IS NULL)
  3. Vector cosine search on chunk-level embeddings (chunk_id IS NOT NULL),
     max-pooled per issue so long issues get credit for matching paragraphs
  4. All three lists fused with RRF (k=60)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

import psycopg
from rank_bm25 import BM25Okapi

from app.core.db import get_conn
from app.retrieval.embedder import embed


RRF_K = 60
DEFAULT_TOP_K = 100
DEFAULT_FINAL_N = 20


@dataclass
class RetrievalHit:
    issue_id: UUID
    title: str
    url: str
    text_full: str
    score: float
    source_scores: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# BM25 index (in-memory, built on demand)
# ---------------------------------------------------------------------------

class BM25Index:
    """In-memory BM25 over issues.text_full loaded from PostgreSQL."""

    def __init__(self) -> None:
        self._corpus_ids: list[UUID] = []
        self._titles: list[str] = []
        self._urls: list[str] = []
        self._texts: list[str] = []
        self._tokenized: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    @property
    def size(self) -> int:
        return len(self._corpus_ids)

    def build(self, conn: psycopg.Connection, repo_filter: str | None = None) -> None:
        """Load issues from DB and build the BM25 index."""
        query = """
            SELECT id, title, url, text_full
            FROM issues
            WHERE text_full IS NOT NULL AND text_full != ''
        """
        params: list = []
        if repo_filter:
            owner, _, name = repo_filter.partition("/")
            if owner and name:
                query += " AND repo_owner = %s AND repo_name = %s"
                params.extend([owner.strip(), name.strip()])
        with conn.cursor() as cur:
            cur.execute(query, params or None)
            rows = cur.fetchall()

        self._corpus_ids = []
        self._titles = []
        self._urls = []
        self._texts = []
        self._tokenized = []
        for row in rows:
            self._corpus_ids.append(row[0])
            self._titles.append(row[1] or "")
            self._urls.append(row[2] or "")
            text_full = row[3] or ""
            self._texts.append(text_full)
            self._tokenized.append(text_full.lower().split())

        if self._tokenized:
            self._bm25 = BM25Okapi(self._tokenized)
        else:
            self._bm25 = None

    def query(self, text: str, top_k: int = DEFAULT_TOP_K) -> list[tuple[UUID, float, str, str, str]]:
        """Return top-K (issue_id, bm25_score, title, url, text_full)."""
        if self._bm25 is None or not text.strip():
            return []
        tokens = text.lower().split()
        scores = self._bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for i in top_indices:
            if scores[i] <= 0:
                continue
            results.append((
                self._corpus_ids[i],
                float(scores[i]),
                self._titles[i],
                self._urls[i],
                self._texts[i],
            ))
        return results


_bm25_cache: dict[str | None, BM25Index] = {}


def get_bm25_index(conn: psycopg.Connection, repo_filter: str | None = None, rebuild: bool = False) -> BM25Index:
    """Return a cached BM25 index keyed by repo_filter (rebuild if missing or forced)."""
    global _bm25_cache
    if repo_filter not in _bm25_cache or rebuild:
        idx = BM25Index()
        idx.build(conn, repo_filter=repo_filter)
        _bm25_cache[repo_filter] = idx
    return _bm25_cache[repo_filter]


# ---------------------------------------------------------------------------
# Vector retrieval (pgvector)
# ---------------------------------------------------------------------------

def _repo_filter_clause(repo_filter: str | None, params: list, table_alias: str = "e") -> str:
    """Append repo owner/name filter clause and extend params in-place. Returns SQL fragment."""
    if not repo_filter:
        return ""
    owner, _, name = repo_filter.partition("/")
    if owner and name:
        params.extend([owner.strip(), name.strip()])
        return f" AND {table_alias}.repo_owner = %s AND {table_alias}.repo_name = %s"
    return ""


def vector_search(
    conn: psycopg.Connection,
    query_embedding: list[float],
    top_k: int = DEFAULT_TOP_K,
    repo_filter: str | None = None,
) -> list[tuple[UUID, float, str, str, str]]:
    """Return top-K (issue_id, cosine_similarity, title, url, text_full) — issue-level only."""
    vec_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
    params: list = [vec_str]
    repo_clause = _repo_filter_clause(repo_filter, params)
    query = f"""
        SELECT e.issue_id,
               1 - (e.embedding <=> %s::vector) AS similarity,
               i.title,
               i.url,
               i.text_full
        FROM issue_embeddings e
        JOIN issues i ON i.id = e.issue_id
        WHERE e.chunk_id IS NULL{repo_clause}
        ORDER BY e.embedding <=> %s::vector
        LIMIT %s
    """
    params.extend([vec_str, top_k])
    with conn.cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()
    return [(r[0], float(r[1]), r[2] or "", r[3] or "", r[4] or "") for r in rows]


def vector_search_chunks(
    conn: psycopg.Connection,
    query_embedding: list[float],
    top_k: int = DEFAULT_TOP_K,
    repo_filter: str | None = None,
) -> list[tuple[UUID, float, str, str, str]]:
    """Search chunk-level embeddings; max-pools per issue_id before returning.

    Returns top-K (issue_id, best_chunk_similarity, title, url, text_full) deduplicated
    by issue, keeping the highest cosine score across all chunks of that issue.
    """
    vec_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
    params: list = [vec_str]
    repo_clause = _repo_filter_clause(repo_filter, params)
    # Fetch more raw rows than top_k so that after dedup we still have top_k unique issues.
    raw_limit = top_k * 5
    query = f"""
        SELECT e.issue_id,
               1 - (e.embedding <=> %s::vector) AS similarity,
               i.title,
               i.url,
               i.text_full
        FROM issue_embeddings e
        JOIN issues i ON i.id = e.issue_id
        WHERE e.chunk_id IS NOT NULL{repo_clause}
        ORDER BY e.embedding <=> %s::vector
        LIMIT %s
    """
    params.extend([vec_str, raw_limit])
    with conn.cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    # Max-pool: keep the highest similarity across all chunks for each issue.
    best: dict[UUID, tuple[float, str, str, str]] = {}
    for issue_id, sim, title, url, text_full in rows:
        if issue_id not in best or sim > best[issue_id][0]:
            best[issue_id] = (sim, title or "", url or "", text_full or "")

    ranked = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)[:top_k]
    return [(uid, data[0], data[1], data[2], data[3]) for uid, data in ranked]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    *ranked_lists: list[tuple[UUID, float, str, str, str]],
    k: int = RRF_K,
) -> dict[UUID, float]:
    """Compute RRF scores across multiple ranked lists.
    Each list is [(issue_id, score, title, url, text_full), ...] in descending score order.
    Returns {issue_id: rrf_score}.
    """
    scores: dict[UUID, float] = {}
    for ranked in ranked_lists:
        for rank, (issue_id, *_rest) in enumerate(ranked, start=1):
            scores[issue_id] = scores.get(issue_id, 0.0) + 1.0 / (k + rank)
    return scores


# ---------------------------------------------------------------------------
# Hybrid search (public entry point)
# ---------------------------------------------------------------------------

def hybrid_search(
    query_text: str,
    top_k: int = DEFAULT_TOP_K,
    final_n: int = DEFAULT_FINAL_N,
    repo_filter: str | None = None,
    conn: psycopg.Connection | None = None,
    query_embedding: list[float] | None = None,
) -> list[RetrievalHit]:
    """Run BM25 + issue-level vector + chunk-level vector search, fuse with RRF.

    Args:
        query_text:      The search query.
        top_k:           Number of candidates per retrieval signal before fusion.
        final_n:         Number of hits to return after fusion.
        repo_filter:     Optional "owner/name" string to scope search to one repo.
        conn:            Optional existing DB connection (will open one if None).
        query_embedding: Optional pre-computed query embedding; avoids a redundant
                         embed() call when the caller already has the vector.

    Returns:
        List of RetrievalHit sorted by RRF score descending.
    """
    should_close = conn is None
    if conn is None:
        ctx = get_conn()
        conn = ctx.__enter__()
    else:
        ctx = None

    try:
        # Embed once — reuse for both issue-level and chunk-level vector search.
        query_vec = query_embedding if query_embedding is not None else embed(query_text)

        # BM25
        bm25_idx = get_bm25_index(conn, repo_filter=repo_filter)
        bm25_results = bm25_idx.query(query_text, top_k=top_k)

        # Issue-level vector search
        vec_results = vector_search(conn, query_vec, top_k=top_k, repo_filter=repo_filter)

        # Chunk-level vector search (max-pooled per issue)
        chunk_results = vector_search_chunks(conn, query_vec, top_k=top_k, repo_filter=repo_filter)

        # RRF merge across all three signals
        rrf_scores = reciprocal_rank_fusion(bm25_results, vec_results, chunk_results)

        # Build metadata and per-signal score maps
        meta: dict[UUID, tuple[str, str, str]] = {}
        bm25_score_map: dict[UUID, float] = {}
        vec_score_map: dict[UUID, float] = {}
        chunk_score_map: dict[UUID, float] = {}

        for issue_id, score, title, url, text_full in bm25_results:
            meta[issue_id] = (title, url, text_full)
            bm25_score_map[issue_id] = score
        for issue_id, score, title, url, text_full in vec_results:
            meta.setdefault(issue_id, (title, url, text_full))
            vec_score_map[issue_id] = score
        for issue_id, score, title, url, text_full in chunk_results:
            meta.setdefault(issue_id, (title, url, text_full))
            chunk_score_map[issue_id] = score

        sorted_ids = sorted(rrf_scores, key=lambda uid: rrf_scores[uid], reverse=True)[:final_n]

        hits = []
        for uid in sorted_ids:
            title, url, text_full = meta.get(uid, ("", "", ""))
            hits.append(RetrievalHit(
                issue_id=uid,
                title=title,
                url=url,
                text_full=text_full,
                score=rrf_scores[uid],
                source_scores={
                    "bm25": bm25_score_map.get(uid, 0.0),
                    "vector": vec_score_map.get(uid, 0.0),
                    "chunk_vector": chunk_score_map.get(uid, 0.0),
                    "rrf": rrf_scores[uid],
                },
            ))
        return hits
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)
