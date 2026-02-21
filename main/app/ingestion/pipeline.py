"""
Ingestion pipeline: after sync + preprocessing, compute embeddings and insert into issue_embeddings.
Run after sync_repos or as a separate "index" step.
"""

from uuid import UUID

from app.core.db import (
    get_conn,
    get_issues_for_embedding,
    delete_issue_embeddings,
    insert_issue_embedding,
)
from app.retrieval.embedder import embed_batch

BATCH_SIZE = 32


def run_index_embeddings() -> tuple[int, int]:
    """
    For each issue with text_full: compute embedding for text_full and for each chunk;
    delete existing embeddings for the issue, then insert new ones.
    Returns (issues_processed, embeddings_written).
    """
    issues_processed = 0
    embeddings_written = 0
    with get_conn() as conn:
        rows = get_issues_for_embedding(conn)
        for issue_id, repo_owner, repo_name, text_full, chunks in rows:
            texts: list[str] = []
            keys: list[tuple[UUID, str | None]] = []
            if text_full:
                texts.append(text_full)
                keys.append((issue_id, None))
            for chunk_id, content in chunks:
                if content:
                    texts.append(content)
                    keys.append((issue_id, chunk_id))
            if not texts:
                continue
            delete_issue_embeddings(conn, issue_id)
            for i in range(0, len(texts), BATCH_SIZE):
                batch_texts = texts[i : i + BATCH_SIZE]
                batch_keys = keys[i : i + BATCH_SIZE]
                vecs = embed_batch(batch_texts)
                for (oid, cid), vec in zip(batch_keys, vecs):
                    insert_issue_embedding(conn, oid, vec, repo_owner, repo_name, chunk_id=cid)
                    embeddings_written += 1
            issues_processed += 1
        conn.commit()
    return (issues_processed, embeddings_written)
