"""
Fine-tune a SetFit urgency classifier from the HuggingFace Hub dataset.

Usage:
  python scripts/train_setfit.py
  python scripts/train_setfit.py --dataset owner/repo --output models/urgency-setfit --epochs 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ml.label_schema import URGENCY_LABELS

_DEFAULT_DATASET = "shinena-xiang/dev-issue-urgency-classification-ds"
_DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "models" / "urgency-setfit"


def _detect_text_col(features: dict) -> str:
    for candidate in ("text_full", "text", "title", "body"):
        if candidate in features:
            return candidate
    raise ValueError(f"No text column found. Available columns: {list(features)}")


def _detect_label_col(features: dict) -> str:
    for candidate in ("urgency", "label", "labels"):
        if candidate in features:
            return candidate
    raise ValueError(f"No label column found. Available columns: {list(features)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SetFit urgency classifier")
    parser.add_argument("--dataset", default=_DEFAULT_DATASET)
    parser.add_argument("--output", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=20,
                        help="Sentence pairs generated per class during contrastive training")
    args = parser.parse_args()

    try:
        import torch
        from datasets import load_dataset
        from setfit import SetFitModel, Trainer, TrainingArguments
    except ImportError as exc:
        print(f"Missing dependency: {exc}\nRun: pip install setfit datasets", file=sys.stderr)
        sys.exit(1)

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    print(f"Loading dataset: {args.dataset}")
    ds = load_dataset(args.dataset)
    print(f"  splits: {list(ds.keys())}")

    train_split = ds.get("train") or ds[list(ds.keys())[0]]
    eval_split = (
        ds.get("test")
        or ds.get("validation")
        or train_split.select(range(max(1, int(len(train_split) * 0.1))))
    )

    features = train_split.features
    print(f"  columns: {list(features)}")

    text_col = _detect_text_col(features)
    label_col = _detect_label_col(features)
    print(f"  text='{text_col}', label='{label_col}'")

    if text_col != "text":
        train_split = train_split.rename_column(text_col, "text")
        eval_split = eval_split.rename_column(text_col, "text")
    if label_col != "label":
        train_split = train_split.rename_column(label_col, "label")
        eval_split = eval_split.rename_column(label_col, "label")

    def _normalize_labels(batch):
        batch["label"] = [str(lbl).lower().strip() for lbl in batch["label"]]
        return batch

    train_split = train_split.map(_normalize_labels, batched=True)
    eval_split = eval_split.map(_normalize_labels, batched=True)

    print(f"  train={len(train_split)}  eval={len(eval_split)}")
    print(f"  labels in data: {sorted(set(train_split['label']))}")

    model = SetFitModel.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2",
        labels=URGENCY_LABELS,
    )
    model.to(device)

    training_args = TrainingArguments(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        num_iterations=args.iterations,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_split,
        eval_dataset=eval_split,
        metric="f1",
    )

    print("Training…")
    trainer.train()

    metrics = trainer.evaluate()
    print(f"Eval metrics: {metrics}")

    output_dir = Path(args.output) if args.output else _DEFAULT_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    print(f"Saved → {output_dir}")


if __name__ == "__main__":
    main()
