CREATE EXTENSION IF NOT EXISTS vector;

ALTER TABLE issues
ADD COLUMN IF NOT EXISTS embedding vector(384);

-- HNSW index for cosine similarity (typical for normalized embeddings)
CREATE INDEX IF NOT EXISTS ix_issues_embedding_hnsw
ON issues
USING hnsw (embedding vector_cosine_ops);

COMMENT ON COLUMN issues.embedding IS 'Optional: embedding vector(384) for semantic search (Phase 2)';
