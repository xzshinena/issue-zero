"""
Fine-tune SetFit classifiers from HuggingFace Hub datasets.

Can train a single task or all four tasks in one run.

Usage:
  # Single task (default: urgency)
  python scripts/train_setfit.py
  python scripts/train_setfit.py --task urgency --dataset owner/ds --epochs 2

  # All four tasks at once (each needs its own --dataset-* arg or a combined dataset)
  python scripts/train_setfit.py --task all

  # Explicit per-task datasets
  python scripts/train_setfit.py --task issue_type --dataset owner/issue-type-ds
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ml.label_schema import (
    ACTION_LABELS,
    IS_REGRESSION_LABELS,
    ISSUE_TYPE_LABELS,
    URGENCY_LABELS,
)

_TASK_LABELS = {
    "urgency": URGENCY_LABELS,
    "issue_type": ISSUE_TYPE_LABELS,
    "action_recommendation": ACTION_LABELS,
    "is_regression": IS_REGRESSION_LABELS,
}

_DEFAULT_DATASETS = {
    "urgency": "shinena-xiang/dev-issue-urgency-classification-ds",
    "issue_type": None,
    "action_recommendation": None,
    "is_regression": None,
}

_MODELS_ROOT = Path(__file__).resolve().parent.parent / "models"


def _detect_text_col(features: dict) -> str:
    for candidate in ("text_full", "text", "title", "body"):
        if candidate in features:
            return candidate
    raise ValueError(f"No text column found. Available columns: {list(features)}")


def _detect_label_col(features: dict, task: str) -> str:
    # Prefer an exact task-name column, then generic "label"/"labels"
    for candidate in (task, "label", "labels"):
        if candidate in features:
            return candidate
    raise ValueError(f"No label column found for task '{task}'. Available columns: {list(features)}")


def _train_task(task: str, dataset_name: str, output_dir: Path, args: argparse.Namespace) -> None:
    labels = _TASK_LABELS[task]

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
    print(f"[{task}] device: {device}")

    print(f"[{task}] loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name)
    print(f"[{task}]   splits: {list(ds.keys())}")

    train_split = ds.get("train") or ds[list(ds.keys())[0]]
    eval_split = (
        ds.get("test")
        or ds.get("validation")
        or train_split.select(range(max(1, int(len(train_split) * 0.1))))
    )

    features = train_split.features
    print(f"[{task}]   columns: {list(features)}")

    text_col = _detect_text_col(features)
    label_col = _detect_label_col(features, task)
    print(f"[{task}]   text='{text_col}', label='{label_col}'")

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

    print(f"[{task}]   train={len(train_split)}  eval={len(eval_split)}")
    print(f"[{task}]   labels in data: {sorted(set(train_split['label']))}")

    model = SetFitModel.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2",
        labels=labels,
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

    print(f"[{task}] training…")
    trainer.train()

    metrics = trainer.evaluate()
    print(f"[{task}] eval metrics: {metrics}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    print(f"[{task}] saved → {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SetFit classifiers")
    parser.add_argument(
        "--task",
        default="urgency",
        choices=list(_TASK_LABELS) + ["all"],
        help="Task to train, or 'all' to train every task that has a --dataset.",
    )
    parser.add_argument("--dataset", default=None,
                        help="HuggingFace dataset for the chosen task.")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: models/<task>-setfit/).")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=20,
                        help="Sentence pairs generated per class during contrastive training.")
    args = parser.parse_args()

    tasks_to_train: list[str] = list(_TASK_LABELS) if args.task == "all" else [args.task]

    for task in tasks_to_train:
        dataset_name = args.dataset or _DEFAULT_DATASETS.get(task)
        if not dataset_name:
            print(f"[{task}] no dataset specified — skipping. "
                  f"Pass --dataset owner/repo or add a default in _DEFAULT_DATASETS.")
            continue
        output_dir = (
            Path(args.output) if (args.output and len(tasks_to_train) == 1)
            else _MODELS_ROOT / f"{task}-setfit"
        )
        _train_task(task, dataset_name, output_dir, args)


if __name__ == "__main__":
    main()
