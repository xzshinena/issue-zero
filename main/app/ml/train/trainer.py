"""Train per-task LogisticRegression classifiers and save as joblib artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from app.ml.label_schema import (
    ACTION_LABELS,
    IS_REGRESSION_LABELS,
    ISSUE_TYPE_LABELS,
    URGENCY_LABELS,
)
from app.ml.train.feature_extractor import extract_features

MODELS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models"

_TASK_LABELS: dict[str, list[str]] = {
    "urgency": URGENCY_LABELS,
    "issue_type": ISSUE_TYPE_LABELS,
    "action_recommendation": ACTION_LABELS,
    "is_regression": IS_REGRESSION_LABELS,
}


# ---------------------------------------------------------------------------
# LabeledClassifier — must be at module level so joblib can pickle it
# ---------------------------------------------------------------------------

class LabeledClassifier:
    """LogisticRegression wrapper that embeds raw text before inference."""

    def __init__(self, task_name: str, clf: Any, le: LabelEncoder) -> None:
        self.task_name = task_name
        self.clf = clf
        self.le = le

    def predict(self, texts: list[str]) -> list[str]:
        X = extract_features(list(texts))
        return list(self.le.inverse_transform(self.clf.predict(X)))

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        X = extract_features(list(texts))
        return self.clf.predict_proba(X)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _extract_label(record: dict, task: str) -> str | None:
    val = record.get(task)
    if val is None:
        return None
    if task == "is_regression":
        return "true" if str(val).lower() in ("true", "1", "yes") else "false"
    return str(val).strip().lower() or None


def train_task(
    task_name: str,
    texts: list[str],
    labels: list[str],
) -> LabeledClassifier:
    """Fit a LabeledClassifier for one task and return it."""
    le = LabelEncoder()
    known = _TASK_LABELS.get(task_name, [])
    if known:
        le.fit(known)
    else:
        le.fit(list(set(labels)))

    y = le.transform(labels)
    X = extract_features(texts)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X, y)

    return LabeledClassifier(task_name, clf, le)


def train_all(
    records: list[dict],
    models_dir: str | Path | None = None,
    use_registry: bool = True,
) -> dict[str, LabeledClassifier]:
    """Train all four classifiers from a list of labeled records.

    When use_registry=True (default), saves each model as a new version under
    models_dir/<task>/v<N>/model.joblib using the model registry.
    Falls back to the flat models_dir/<task>.joblib layout when use_registry=False.
    Returns a dict of task_name -> LabeledClassifier.
    """
    out_dir = Path(models_dir) if models_dir else MODELS_DIR

    trained: dict[str, LabeledClassifier] = {}
    for task in _TASK_LABELS:
        pairs = [
            (r.get("text_full", ""), _extract_label(r, task))
            for r in records
            if _extract_label(r, task) is not None
        ]
        if not pairs:
            print(f"[{task}] no labeled examples — skipping")
            continue

        texts, labels = zip(*pairs)
        model = train_task(task, list(texts), list(labels))

        if use_registry:
            from app.ml.model_registry import save_model  # noqa: PLC0415
            version = save_model(
                task, model, model_type="sklearn",
                extra={"n_examples": len(texts)},
                models_dir=out_dir,
            )
            print(f"[{task}] trained on {len(texts)} examples → {out_dir}/{task}/v{version}/")
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"{task}.joblib"
            joblib.dump(model, path)
            print(f"[{task}] trained on {len(texts)} examples → {path}")

        trained[task] = model

    return trained
