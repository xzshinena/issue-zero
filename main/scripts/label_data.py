"""
Normalize scraped GitHub issues JSON to training-ready JSONL.

Input: JSON array (or JSONL) of GitHub issue objects, optionally with
       user-added label fields (urgency, issue_type, action_recommendation,
       is_regression).

Output: JSONL where each line has:
  text_full, urgency?, issue_type?, action_recommendation?, is_regression?

Usage:
  python scripts/label_data.py --input data/issues.json --output data/labeled.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def normalize(record: dict) -> dict | None:
    """Map a raw GitHub issue dict to a normalized training record."""
    title = (record.get("title") or "").strip()
    body = (record.get("body") or "").strip()
    if not title and not body:
        return None

    text_full = f"{title}\n\n{body}".strip() if body else title

    out: dict = {"text_full": text_full}

    labels_raw = record.get("labels") or []
    if isinstance(labels_raw, list):
        label_names = []
        for lb in labels_raw:
            if isinstance(lb, dict):
                label_names.append(lb.get("name", ""))
            elif isinstance(lb, str):
                label_names.append(lb)
        out["github_labels"] = [n for n in label_names if n]

    for field in ("urgency", "issue_type", "action_recommendation", "is_regression"):
        if field in record and record[field] is not None:
            out[field] = record[field]

    return out


def _load_input(path: Path) -> list[dict]:
    """Read JSON array or JSONL from path."""
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("["):
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON array: {e}", file=sys.stderr)
            sys.exit(1)

    records = []
    for i, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Skipping record {i}: {e}", file=sys.stderr)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize GitHub issues JSON to training JSONL")
    parser.add_argument("--input", required=True, help="Path to input JSON or JSONL")
    parser.add_argument("--output", required=True, help="Path to output JSONL")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw = _load_input(in_path)
    print(f"Loaded {len(raw)} records from {in_path}")

    written = 0
    skipped = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, record in enumerate(raw, start=1):
            try:
                normalized = normalize(record)
            except Exception as e:
                print(f"Skipping record {i}: {e}", file=sys.stderr)
                skipped += 1
                continue
            if normalized is None:
                skipped += 1
                continue
            f.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} records to {out_path} ({skipped} skipped)")


if __name__ == "__main__":
    main()
