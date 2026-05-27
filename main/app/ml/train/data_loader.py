"""Load labeled training data from JSONL and return train/val splits."""

from __future__ import annotations

import json
from pathlib import Path

_LABEL_FIELDS = ("urgency", "issue_type", "action_recommendation", "is_regression")


def _is_labeled(record: dict) -> bool:
    return any(field in record and record[field] is not None for field in _LABEL_FIELDS)


def load_records(path: str | Path) -> list[dict]:
    """Read JSONL; return only rows that have at least one label field."""
    records = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping record {i}: {e}")
                continue
            if _is_labeled(record):
                records.append(record)
    return records


def train_val_split(
    records: list[dict],
    val_frac: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Shuffle and split records into (train, val)."""
    import random

    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    split = max(1, int(len(shuffled) * (1 - val_frac)))
    return shuffled[:split], shuffled[split:]


def load_split(
    path: str | Path,
    val_frac: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Load JSONL and return (train_records, val_records)."""
    records = load_records(path)
    if not records:
        raise ValueError(f"No labeled records found in {path}")
    return train_val_split(records, val_frac=val_frac, seed=seed)
