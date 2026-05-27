"""Tests for scripts/label_data.py normalize + main."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from label_data import normalize, _load_input, main


def test_normalize_valid():
    rec = {"title": "Bug: crash", "body": "Crashes on startup", "urgency": "critical_bug"}
    out = normalize(rec)
    assert out is not None
    assert "Bug: crash" in out["text_full"]
    assert out["urgency"] == "critical_bug"


def test_normalize_missing_body():
    rec = {"title": "Short issue"}
    out = normalize(rec)
    assert out is not None
    assert out["text_full"] == "Short issue"


def test_normalize_no_title_no_body_returns_none():
    assert normalize({"title": "", "body": ""}) is None


def test_main_skips_bad_json(tmp_path):
    in_path = tmp_path / "issues.jsonl"
    in_path.write_text(
        '{"title": "valid", "body": "ok"}\n'
        "NOT_JSON\n"
        '{"title": "also valid", "body": "fine"}\n',
        encoding="utf-8",
    )
    out_path = tmp_path / "out.jsonl"

    sys.argv = ["label_data.py", "--input", str(in_path), "--output", str(out_path)]
    main()

    lines = [l for l in out_path.read_text().splitlines() if l.strip()]
    assert len(lines) == 2
