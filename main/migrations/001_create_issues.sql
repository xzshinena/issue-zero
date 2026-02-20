-- Issues table (canonical schema)
-- Run from project root with: psql $DATABASE_URL -f migrations/001_create_issues.sql

CREATE TABLE IF NOT EXISTS issues (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source TEXT NOT NULL CHECK (source IN ('github', 'gitlab')),
    repo_owner TEXT NOT NULL,
    repo_name TEXT NOT NULL,
    issue_number INTEGER NOT NULL,
    title TEXT NOT NULL,
    body TEXT NOT NULL DEFAULT '',
    body_plain TEXT NOT NULL DEFAULT '',
    state TEXT NOT NULL CHECK (state IN ('open', 'closed')),
    labels JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    closed_at TIMESTAMPTZ,
    url TEXT NOT NULL DEFAULT '',
    text_full TEXT NOT NULL DEFAULT '',
    chunk_ids TEXT[],
    v2_issue_type TEXT,
    v2_severity TEXT,
    v2_component TEXT,
    CONSTRAINT uq_issue_source_repo_number UNIQUE (source, repo_owner, repo_name, issue_number)
);

CREATE INDEX IF NOT EXISTS ix_issues_repo ON issues (source, repo_owner, repo_name);
CREATE INDEX IF NOT EXISTS ix_issues_updated_at ON issues (updated_at DESC);
CREATE INDEX IF NOT EXISTS ix_issues_state ON issues (state);
