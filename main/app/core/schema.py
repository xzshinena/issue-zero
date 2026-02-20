"""Canonical issue schema and API schema definitions."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class Issue(BaseModel):
    """Canonical issue model for storage and retrieval."""

    id: str | None = None  # UUID set by DB; omit on insert
    source: Literal["github", "gitlab"]
    repo_owner: str
    repo_name: str
    issue_number: int
    title: str
    body: str = ""
    body_plain: str = ""
    state: Literal["open", "closed"]
    labels: list[str] = Field(default_factory=list)  # stored as JSONB
    created_at: datetime
    updated_at: datetime
    closed_at: datetime | None = None
    url: str = ""
    text_full: str = ""  # title + " " + body_plain (built in preprocessing)

    # Optional (chunking / Phase 2)
    chunk_ids: list[str] | None = None
    v2_issue_type: str | None = None
    v2_severity: str | None = None
    v2_component: str | None = None
