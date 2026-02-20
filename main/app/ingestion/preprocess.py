"""Preprocessing: clean template blocks, build text_full, chunk long issues."""

import re
from uuid import UUID

import psycopg

from app.core.schema import Issue


# ~512 tokens ≈ 2048 chars (rough 4 chars/token)
CHUNK_MAX_CHARS = 2048
CHUNK_OVERLAP_CHARS = 128

# Default patterns: section header + content until next ## or end (non-greedy)
DEFAULT_TEMPLATE_PATTERNS: list[tuple[str, str]] = [
    (r"(?ms)\n##\s*Environment\s*\n.*?(?=\n##|\Z)", ""),
    (r"(?ms)\n##\s*Steps to reproduce\s*\n.*?(?=\n##|\Z)", ""),
    (r"(?ms)\n##\s*What happened\s*\n.*?(?=\n##|\Z)", ""),
    (r"(?ms)\n##\s*Expected behavior\s*\n.*?(?=\n##|\Z)", ""),
    (r"(?ms)\n\*\*Describe the bug\*\*\s*\n.*?(?=\n##|\n\*\*|\Z)", ""),
    (r"(?ms)\n<!--.*?-->\s*", " "),  # HTML comments (e.g. template instructions)
]

# Per-repo overrides: "owner/repo" -> list of (pattern, repl)
REPO_TEMPLATE_PATTERNS: dict[str, list[tuple[str, str]]] = {}


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace to single space and strip."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def clean_body_plain(
    body_plain: str,
    repo_key: str | None = None,
    extra_patterns: list[tuple[str, str]] | None = None,
) -> str:
    """
    Remove common issue template blocks from body_plain.
    repo_key: optional "owner/repo" for per-repo patterns.
    extra_patterns: optional (regex, replacement) list.
    """
    if not body_plain:
        return ""
    s = body_plain
    patterns = list(DEFAULT_TEMPLATE_PATTERNS)
    if repo_key and repo_key in REPO_TEMPLATE_PATTERNS:
        patterns = list(REPO_TEMPLATE_PATTERNS[repo_key])
    if extra_patterns:
        patterns = patterns + list(extra_patterns)
    for pattern, repl in patterns:
        s = re.sub(pattern, repl, s)
    return normalize_whitespace(s)


def build_text_full(
    title: str,
    body_plain: str,
    labels: list[str] | None = None,
    prepend_labels: bool = True,
) -> str:
    """
    Build text_full = title + " " + normalize_whitespace(body_plain).
    If prepend_labels, prefix with label names for BM25: " ".join(labels) + " " + rest.
    """
    body_norm = normalize_whitespace(body_plain)
    if prepend_labels and labels:
        prefix = " ".join(labels) + " "
        return normalize_whitespace(prefix + title + " " + body_norm)
    return normalize_whitespace(title + " " + body_norm)


def _estimate_tokens(text: str) -> int:
    """Rough token count (~4 chars per token)."""
    return max(1, len(text) // 4)


def chunk_text(
    text: str,
    max_chars: int = CHUNK_MAX_CHARS,
    overlap_chars: int = CHUNK_OVERLAP_CHARS,
) -> list[dict]:
    """
    Split text into chunks: try paragraph boundaries first, then sliding window with overlap.
    Returns list of {"index": int, "content": str}.
    """
    if not text or _estimate_tokens(text) <= max_chars // 4:
        return []

    chunks: list[dict] = []
    # Split by paragraphs (double newline or single newline)
    paras = re.split(r"\n\s*\n", text)
    current = ""
    for p in paras:
        p = p.strip()
        if not p:
            continue
        if len(current) + len(p) + 1 <= max_chars:
            current = (current + " " + p).strip() if current else p
        else:
            if current:
                chunks.append({"content": current})
            if _estimate_tokens(p) * 4 <= max_chars:
                current = p
            else:
                # Sliding window with overlap within this paragraph
                start = 0
                while start < len(p):
                    end = start + max_chars
                    if end >= len(p):
                        chunk = p[start:].strip()
                        if chunk:
                            chunks.append({"content": chunk})
                        break
                    # Break at word boundary if possible
                    break_at = p.rfind(" ", start, end + 1)
                    if break_at > start:
                        end = break_at
                    chunk = p[start:end].strip()
                    if chunk:
                        chunks.append({"content": chunk})
                    start = end - overlap_chars
                current = ""
    if current:
        chunks.append({"content": current})

    for i, c in enumerate(chunks):
        c["index"] = i
    return chunks


def preprocess_issue(
    issue: Issue,
    repo_key: str | None = None,
    prepend_labels: bool = True,
    chunk_threshold_chars: int = CHUNK_MAX_CHARS,
) -> tuple[str, list[str] | None, list[dict]]:
    """
    Run clean, build text_full, and optionally chunk.
    Returns (text_full, chunk_ids or None, chunks list with index/content).
    chunk_ids are placeholders "PLACEHOLDER#0", "PLACEHOLDER#1"; caller replaces with issue_id after upsert.
    """
    cleaned = clean_body_plain(issue.body_plain or "", repo_key=repo_key)
    text_full = build_text_full(
        issue.title or "",
        cleaned,
        labels=issue.labels or [],
        prepend_labels=prepend_labels,
    )
    chunks: list[dict] = []
    if len(text_full) > chunk_threshold_chars:
        chunks = chunk_text(text_full, max_chars=chunk_threshold_chars)
    chunk_ids = [f"PLACEHOLDER#{c['index']}" for c in chunks] if chunks else None
    return (text_full, chunk_ids, chunks)


def run_after_upsert(
    conn: psycopg.Connection,
    issue_id: UUID,
    issue: Issue,
    repo_key: str | None = None,
    prepend_labels: bool = True,
) -> None:
    """
    After upsert: run preprocessing, update issues.text_full and issues.chunk_ids,
    and insert into issue_chunks. Deletes existing chunks for this issue first (idempotent re-run).
    """
    text_full, chunk_ids, chunks = preprocess_issue(
        issue, repo_key=repo_key, prepend_labels=prepend_labels
    )
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE issues
            SET text_full = %s, chunk_ids = %s
            WHERE id = %s
            """,
            (text_full, chunk_ids, issue_id),
        )
        # Replace placeholder with real issue_id for chunk_id
        id_str = str(issue_id)
        cur.execute(
            "DELETE FROM issue_chunks WHERE issue_id = %s",
            (issue_id,),
        )
        for c in chunks:
            chunk_id = f"{id_str}#{c['index']}"
            cur.execute(
                """
                INSERT INTO issue_chunks (issue_id, chunk_index, chunk_id, content)
                VALUES (%s, %s, %s, %s)
                """,
                (issue_id, c["index"], chunk_id, c["content"]),
            )
    # conn.commit is left to caller (same transaction as upsert)
