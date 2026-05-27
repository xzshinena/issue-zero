"""
Train all four classifiers from a labeled JSONL file.

Usage:
  python scripts/train_classifiers.py --data data/labeled.jsonl
  python scripts/train_classifiers.py --data data/labeled.jsonl --models-dir models/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ml.train.data_loader import load_split
from app.ml.train.trainer import train_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Train issue classifiers")
    parser.add_argument("--data", required=True, help="Path to labeled JSONL")
    parser.add_argument("--models-dir", default=None, help="Output dir for *.joblib (default: models/)")
    parser.add_argument("--val-frac", type=float, default=0.2, help="Validation fraction")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: {data_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Loading data from {data_path}…")
    train_records, val_records = load_split(data_path, val_frac=args.val_frac)
    print(f"  train={len(train_records)}  val={len(val_records)}")

    print("Training classifiers…")
    trained = train_all(train_records, models_dir=args.models_dir)

    if not trained:
        print("No classifiers trained — check that your data has label fields.", file=sys.stderr)
        sys.exit(1)

    print(f"\nDone. {len(trained)} classifiers saved.")
    if val_records:
        print("Run eval/classification_metrics.py to measure val accuracy.")


if __name__ == "__main__":
    main()
