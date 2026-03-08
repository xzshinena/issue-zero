"""
Classification evaluation: per-task macro-F1 and per-class precision/recall/F1.

Input: a JSONL file where each line is:
  {
    "text": "...",
    "labels": ["bug", ...],        # optional GitHub labels for heuristic classifiers
    "urgency": "high",             # ground truth
    "issue_type": "bug",
    "action_recommendation": "triage",
    "is_regression": false
  }

Usage:
  python eval/classification_metrics.py --eval-file eval/classification_set.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_MAIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MAIN not in sys.path:
    sys.path.insert(0, _MAIN)


TASKS = ["urgency", "issue_type", "action_recommendation", "is_regression"]


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """Compute per-class and macro precision, recall, F1."""
    classes = sorted(set(y_true) | set(y_pred))
    per_class: dict[str, dict] = {}
    macro_p, macro_r, macro_f = 0.0, 0.0, 0.0

    for cls in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = _f1(precision, recall)
        per_class[cls] = {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4), "support": tp + fn}
        macro_p += precision
        macro_r += recall
        macro_f += f1

    n = max(len(classes), 1)
    return {
        "per_class": per_class,
        "macro_precision": round(macro_p / n, 4),
        "macro_recall": round(macro_r / n, 4),
        "macro_f1": round(macro_f / n, 4),
        "num_samples": len(y_true),
    }


def evaluate(eval_entries: list[dict]) -> dict:
    """Run classifiers on eval set and compute metrics per task."""
    from app.ml.classifiers import predict

    results: dict[str, dict] = {}

    for task in TASKS:
        y_true = []
        y_pred = []
        for entry in eval_entries:
            gt = entry.get(task)
            if gt is None:
                continue
            text = entry.get("text", "")
            labels = entry.get("labels", [])
            preds = predict(text, labels=labels)

            if task == "is_regression":
                y_true.append(str(gt).lower())
                y_pred.append(str(preds.is_regression).lower())
            else:
                y_true.append(str(gt))
                y_pred.append(str(getattr(preds, task)))

        if y_true:
            results[task] = compute_metrics(y_true, y_pred)
        else:
            results[task] = {"error": "no labeled samples for this task"}

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate classification quality.")
    parser.add_argument("--eval-file", required=True, help="JSONL file with labeled samples.")
    args = parser.parse_args()

    if not os.path.exists(args.eval_file):
        print(f"error: file not found: {args.eval_file}", file=sys.stderr)
        return 1

    entries = []
    with open(args.eval_file) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    if not entries:
        print("error: no entries", file=sys.stderr)
        return 1

    results = evaluate(entries)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
