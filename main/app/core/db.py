"""Database connection pool and session/transaction helpers."""

from contextlib import contextmanager
from typing import Iterator
from uuid import UUID

import psycopg
from psycopg_pool import ConnectionPool

from app.core.config import Settings, get_settings
from app.core.schema import Issue


_pool: ConnectionPool | None = None


def get_pool(settings: Settings | None = None) -> ConnectionPool:
    """Return the global connection pool; create it if needed."""
    global _pool
    if _pool is None:
        settings = settings or get_settings()
        _pool = ConnectionPool(
            conninfo=settings.effective_database_url,
            min_size=1,
            max_size=10,
        )
    return _pool


def close_pool() -> None:
    """Close the global pool (e.g. on app shutdown)."""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


@contextmanager
def get_conn(settings: Settings | None = None) -> Iterator[psycopg.Connection]:
    """Yield a connection from the pool for the duration of the context."""
    pool = get_pool(settings)
    with pool.connection() as conn:
        yield conn


def _issue_to_row(issue: Issue) -> tuple:
    """Map Issue to DB row (id omitted for insert; used for upsert)."""
    return (
        issue.source,
        issue.repo_owner,
        issue.repo_name,
        issue.issue_number,
        issue.title,
        issue.body,
        issue.body_plain,
        issue.state,
        issue.labels,
        issue.created_at,
        issue.updated_at,
        issue.closed_at,
        issue.url,
        issue.text_full,
        issue.chunk_ids,
        issue.v2_issue_type,
        issue.v2_severity,
        issue.v2_component,
    )


def upsert_issue(conn: psycopg.Connection, issue: Issue) -> UUID:
    """
    Insert or update an issue by (source, repo_owner, repo_name, issue_number).
    Returns the issue id (existing or new).
    """
    row = _issue_to_row(issue)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO issues (
                source, repo_owner, repo_name, issue_number,
                title, body, body_plain, state, labels,
                created_at, updated_at, closed_at, url, text_full,
                chunk_ids, v2_issue_type, v2_severity, v2_component
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb,
                %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (source, repo_owner, repo_name, issue_number)
            DO UPDATE SET
                title = EXCLUDED.title,
                body = EXCLUDED.body,
                body_plain = EXCLUDED.body_plain,
                state = EXCLUDED.state,
                labels = EXCLUDED.labels,
                updated_at = EXCLUDED.updated_at,
                closed_at = EXCLUDED.closed_at,
                url = EXCLUDED.url,
                text_full = EXCLUDED.text_full,
                chunk_ids = EXCLUDED.chunk_ids,
                v2_issue_type = EXCLUDED.v2_issue_type,
                v2_severity = EXCLUDED.v2_severity,
                v2_component = EXCLUDED.v2_component
            RETURNING id
            """,
            row,
        )
        (id_val,) = cur.fetchone()
        return id_val


def get_issues_for_embedding(conn: psycopg.Connection) -> list[tuple[UUID, str, str, str, list[tuple[str, str]]]]:
    """
    Returns list of (issue_id, repo_owner, repo_name, text_full, [(chunk_id, content), ...])
    for all issues that have non-empty text_full.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT i.id, i.repo_owner, i.repo_name, i.text_full,
                   c.chunk_id, c.content
            FROM issues i
            LEFT JOIN issue_chunks c ON c.issue_id = i.id
            WHERE i.text_full IS NOT NULL AND i.text_full != ''
            ORDER BY i.id, c.chunk_index
            """
        )
        rows = cur.fetchall()

    seen: dict[UUID, int] = {}
    out: list[tuple[UUID, str, str, str, list[tuple[str, str]]]] = []
    for row in rows:
        issue_id, repo_owner, repo_name, text_full, chunk_id, content = row
        if issue_id not in seen:
            seen[issue_id] = len(out)
            out.append((issue_id, repo_owner, repo_name, (text_full or "").strip(), []))
        if chunk_id is not None:
            out[seen[issue_id]][4].append((chunk_id, content or ""))
    return out


def delete_issue_embeddings(conn: psycopg.Connection, issue_id: UUID) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM issue_embeddings WHERE issue_id = %s", (issue_id,))


def insert_issue_embedding(
    conn: psycopg.Connection,
    issue_id: UUID,
    embedding: list[float],
    repo_owner: str,
    repo_name: str,
    chunk_id: str | None = None,
) -> None:
    vector_str = "[" + ",".join(str(x) for x in embedding) + "]"
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO issue_embeddings (issue_id, chunk_id, embedding, repo_owner, repo_name)
            VALUES (%s, %s, %s::vector, %s, %s)
            ON CONFLICT (issue_id, chunk_id) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                repo_owner = EXCLUDED.repo_owner,
                repo_name = EXCLUDED.repo_name
            """,
            (issue_id, chunk_id, vector_str, repo_owner, repo_name),
        )


def row_to_issue(row: tuple) -> Issue:
    """Map a DB row to an Issue model. Row must be (id, source, ..., v2_component) in table order (omit embedding if present)."""
    return Issue(
        id=str(row[0]) if row[0] else None,
        source=row[1],
        repo_owner=row[2],
        repo_name=row[3],
        issue_number=row[4],
        title=row[5],
        body=row[6] or "",
        body_plain=row[7] or "",
        state=row[8],
        labels=row[9] or [],
        created_at=row[10],
        updated_at=row[11],
        closed_at=row[12],
        url=row[13] or "",
        text_full=row[14] or "",
        chunk_ids=row[15],
        v2_issue_type=row[16],
        v2_severity=row[17],
        v2_component=row[18],
    )
