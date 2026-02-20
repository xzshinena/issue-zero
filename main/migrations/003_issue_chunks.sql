-- Chunks for long issues (reference to parent issue)
-- Run after 001: psql $DATABASE_URL -f migrations/003_issue_chunks.sql

CREATE TABLE IF NOT EXISTS issue_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    issue_id UUID NOT NULL REFERENCES issues (id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_id TEXT NOT NULL,
    content TEXT NOT NULL,
    CONSTRAINT uq_issue_chunk_id UNIQUE (chunk_id)
);

CREATE INDEX IF NOT EXISTS ix_issue_chunks_issue_id ON issue_chunks (issue_id);

COMMENT ON TABLE issue_chunks IS 'Chunks of issue text_full for long issues (~512+ tokens). chunk_id = issue_id#index';
