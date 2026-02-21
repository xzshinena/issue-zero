-- Optional: use vector(1536) for OpenAI embeddings.
-- Run only if you use embedding_provider=openai. Otherwise keep 004 (vector 384).

-- Drop and recreate to change dimension (pgvector does not support ALTER column type easily)
DROP INDEX IF EXISTS ix_issue_embeddings_hnsw;
DROP INDEX IF EXISTS ix_issue_embeddings_repo;
DROP INDEX IF EXISTS ix_issue_embeddings_issue_id;
DROP TABLE IF EXISTS issue_embeddings;

CREATE TABLE issue_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    issue_id UUID NOT NULL REFERENCES issues (id) ON DELETE CASCADE,
    chunk_id TEXT,
    embedding vector(1536) NOT NULL,
    repo_owner TEXT NOT NULL,
    repo_name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_issue_embedding_issue_chunk UNIQUE (issue_id, chunk_id)
);

CREATE INDEX ix_issue_embeddings_issue_id ON issue_embeddings (issue_id);
CREATE INDEX ix_issue_embeddings_repo ON issue_embeddings (repo_owner, repo_name);
CREATE INDEX ix_issue_embeddings_hnsw ON issue_embeddings USING hnsw (embedding vector_cosine_ops);
