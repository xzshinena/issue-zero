"""Tests for app/ml/train/data_loader.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.ml.train.data_loader import load_records, load_split


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")


def test_load_labeled_only(tmp_path):
    records = [
        {"text_full": "crash", "urgency": "critical_bug"},
        {"text_full": "no labels here"},
        {"text_full": "feature", "issue_type": "feature_request"},
    ]
    p = tmp_path / "data.jsonl"
    _write_jsonl(p, records)
    loaded = load_records(p)
    assert len(loaded) == 2


def test_load_unlabeled_skipped(tmp_path):
    records = [{"text_full": "plain text, no label"}]
    p = tmp_path / "data.jsonl"
    _write_jsonl(p, records)
    assert load_records(p) == []


def test_train_val_split_ratio(tmp_path):
    records = [{"text_full": f"issue {i}", "urgency": "medium"} for i in range(100)]
    p = tmp_path / "data.jsonl"
    _write_jsonl(p, records)
    train, val = load_split(p, val_frac=0.2)
    assert abs(len(train) - 80) <= 2
    assert abs(len(val) - 20) <= 2


def test_empty_file_raises(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="No labeled records"):
        load_split(p)
