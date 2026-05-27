"""Shared pytest fixtures."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


@pytest.fixture
def sample_issues_jsonl(tmp_path: Path) -> Path:
    """Write 10 labeled issue records to a tmp JSONL file."""
    actions = ["triage", "assign_to_area", "need_more_info", "duplicate", "close"]
    records = [
        {
            "text_full": f"Issue {i}: crash on startup with null pointer",
            "urgency": "critical_bug" if i % 3 == 0 else "medium",
            "issue_type": "bug" if i % 2 == 0 else "feature_request",
            "action_recommendation": actions[i % len(actions)],
            "is_regression": i % 4 == 0,
        }
        for i in range(10)
    ]
    out = tmp_path / "labeled.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return out


@pytest.fixture
def trained_models(tmp_path: Path, sample_issues_jsonl: Path):
    """Run train_all on sample data; return (models_dir, trained dict)."""
    from app.ml.train.data_loader import load_records
    from app.ml.train.trainer import train_all

    models_dir = tmp_path / "models"
    records = load_records(sample_issues_jsonl)
    trained = train_all(records, models_dir=models_dir)
    return models_dir, trained


@pytest.fixture
def pg_conn():
    """Yield a real psycopg connection from TEST_DATABASE_URL (skip if not set)."""
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")
    import psycopg
    with psycopg.connect(url) as conn:
        yield conn
