-- issue_embeddings: one row per issue (chunk_id NULL) or per chunk (chunk_id set)
-- Requires: 001_create_issues.sql, 003_issue_chunks.sql, and pgvector (002 or CREATE EXTENSION vector)
-- For OpenAI (1536-dim) run 005_issue_embeddings_1536.sql after this, or alter embedding column.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS issue_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    issue_id UUID NOT NULL REFERENCES issues (id) ON DELETE CASCADE,
    chunk_id TEXT,
    embedding vector(384) NOT NULL,
    repo_owner TEXT NOT NULL,
    repo_name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_issue_embedding_issue_chunk UNIQUE (issue_id, chunk_id)
);

CREATE INDEX IF NOT EXISTS ix_issue_embeddings_issue_id ON issue_embeddings (issue_id);
CREATE INDEX IF NOT EXISTS ix_issue_embeddings_repo ON issue_embeddings (repo_owner, repo_name);

CREATE INDEX IF NOT EXISTS ix_issue_embeddings_hnsw
ON issue_embeddings
USING hnsw (embedding vector_cosine_ops);

COMMENT ON TABLE issue_embeddings IS 'Embeddings for issue text_full (chunk_id NULL) and per chunk (chunk_id set). vector(384) for sentence-transformers. Use 005 for OpenAI 1536';
