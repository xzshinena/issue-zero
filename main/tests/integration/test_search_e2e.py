"""Integration tests — require TEST_DATABASE_URL env var (skipped otherwise)."""

import pytest


@pytest.mark.integration
def test_placeholder(pg_conn):
    """Placeholder: add roundtrip ingest→search tests once pg_conn is wired."""
    assert pg_conn is not None
