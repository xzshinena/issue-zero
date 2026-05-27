"""
Classifier loading and inference for four tasks:
  - urgency              (critical_bug, high, medium, low, enhancement, question)
  - issue_type           (bug, feature_request, docs, refactor, regression, question)
  - action_recommendation(triage, assign_to_area, need_more_info, duplicate, close)
  - is_regression        (True / False)

Prediction priority per task (independent — no "all 4 or nothing"):
  1. SetFit model  — preferred; loaded from models/<task>-setfit/
  2. sklearn LR    — loaded from models/<task>.joblib
  3. Heuristic     — regex patterns; always available as final fallback

When `embedding` is supplied to predict(), the sklearn LR path skips re-encoding
(it calls the classifier head directly). The SetFit path also uses the pre-computed
vector when the embedding dimension matches, avoiding a second encode call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.ml.label_schema import ACTION_LABELS, ISSUE_TYPE_LABELS, URGENCY_LABELS

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

ALL_TASKS = ("urgency", "issue_type", "action_recommendation", "is_regression")

_TASK_LABELS: dict[str, list[str]] = {
    "urgency": URGENCY_LABELS,
    "issue_type": ISSUE_TYPE_LABELS,
    "action_recommendation": ACTION_LABELS,
    "is_regression": ["true", "false"],
}


@dataclass
class Predictions:
    urgency: str
    urgency_confidence: float
    issue_type: str
    issue_type_confidence: float
    action_recommendation: str
    action_confidence: float
    is_regression: bool
    regression_confidence: float


# ---------------------------------------------------------------------------
# Model registry (module-level singletons)
# ---------------------------------------------------------------------------

_trained_models: dict[str, Any] = {}
_models_loaded: bool = False

_setfit_models: dict[str, Any] = {}   # task -> SetFitModel
_setfit_loaded: bool = False

# Module-level so joblib can unpickle LabeledClassifier
_setfit_model = _setfit_models.get("urgency")  # kept for backward compat


def _try_load_setfit() -> bool:
    """Load SetFit models for every task that has a saved model directory.
    Returns True if at least one model loaded successfully.
    """
    global _setfit_models, _setfit_loaded
    if _setfit_loaded:
        return bool(_setfit_models)
    _setfit_loaded = True

    try:
        from setfit import SetFitModel  # noqa: PLC0415
    except ImportError:
        return False

    loaded_any = False
    for task in ALL_TASKS:
        task_dir = MODELS_DIR / f"{task}-setfit"
        if not task_dir.is_dir():
            continue
        try:
            _setfit_models[task] = SetFitModel.from_pretrained(str(task_dir))
            loaded_any = True
        except Exception:
            pass

    return loaded_any


def _try_load_trained() -> bool:
    """Load sklearn joblib models. Returns True if any loaded."""
    global _trained_models, _models_loaded
    if _models_loaded:
        return bool(_trained_models)
    _models_loaded = True

    try:
        import joblib  # noqa: PLC0415
    except ImportError:
        return False

    for task in ALL_TASKS:
        path = MODELS_DIR / f"{task}.joblib"
        if path.exists():
            try:
                _trained_models[task] = joblib.load(path)
            except Exception:
                pass

    return bool(_trained_models)


# ---------------------------------------------------------------------------
# Per-task prediction helpers
# ---------------------------------------------------------------------------

def _predict_task_with_models(
    task: str,
    text: str,
    embedding: list[float] | None,
) -> tuple[str, float] | None:
    """Try SetFit then sklearn LR for a single task.
    Returns (label, confidence) or None if neither is available.
    """
    # --- SetFit path ---
    sf_model = _setfit_models.get(task)
    if sf_model is not None:
        try:
            if embedding is not None:
                vec = np.array([embedding], dtype=np.float32)
                # Use the classification head directly when we have a pre-computed vector.
                # This avoids re-encoding via the SetFit body.
                head = getattr(sf_model, "model_head", None)
                if head is not None and hasattr(head, "predict_proba"):
                    proba = head.predict_proba(vec)[0]
                    label_idx = int(np.argmax(proba))
                    labels = _TASK_LABELS.get(task, [])
                    label = labels[label_idx] if label_idx < len(labels) else str(label_idx)
                    return label, float(proba[label_idx])
            # Fall back to full SetFit predict when no pre-computed embedding.
            pred_label = str(sf_model.predict([text])[0])
            conf = 0.0
            if hasattr(sf_model, "predict_proba"):
                proba = sf_model.predict_proba([text])[0]
                conf = float(np.max(proba))
            return pred_label, conf
        except Exception:
            pass

    # --- sklearn LR path ---
    lr_model = _trained_models.get(task)
    if lr_model is not None:
        try:
            if embedding is not None:
                vec = np.array([embedding])
                pred_raw = lr_model.clf.predict(vec)[0]
                pred_label = str(lr_model.le.inverse_transform([pred_raw])[0])
                conf = float(np.max(lr_model.clf.predict_proba(vec)[0]))
            else:
                preds = lr_model.predict([text])
                pred_label = preds[0]
                proba = lr_model.predict_proba([text])
                conf = float(np.max(proba[0]))
            return pred_label, conf
        except Exception:
            pass

    return None


def _predict_trained(
    text: str,
    embedding: list[float] | None = None,
) -> Predictions | None:
    """Use per-task models (SetFit or LR) for each of the four tasks.
    Returns None only when *no* trained model is available for *any* task
    (which means we should fall through entirely to heuristic).
    """
    _try_load_setfit()
    if not _trained_models:
        _try_load_trained()

    if not _setfit_models and not _trained_models:
        return None

    results: dict[str, tuple[str, float] | None] = {}
    for task in ALL_TASKS:
        results[task] = _predict_task_with_models(task, text, embedding)

    # If every task returned None, fall through to heuristic entirely.
    if all(v is None for v in results.values()):
        return None

    # Fill any missing tasks from heuristic.
    heuristic = _heuristic_predict(text)
    urg_label, urg_conf = results.get("urgency") or (heuristic.urgency, heuristic.urgency_confidence)
    type_label, type_conf = results.get("issue_type") or (heuristic.issue_type, heuristic.issue_type_confidence)
    act_label, act_conf = results.get("action_recommendation") or (heuristic.action_recommendation, heuristic.action_confidence)
    reg_raw, reg_conf = results.get("is_regression") or (str(heuristic.is_regression).lower(), heuristic.regression_confidence)

    return Predictions(
        urgency=urg_label,
        urgency_confidence=urg_conf,
        issue_type=type_label,
        issue_type_confidence=type_conf,
        action_recommendation=act_label,
        action_confidence=act_conf,
        is_regression=str(reg_raw).lower() in ("true", "1", "yes"),
        regression_confidence=reg_conf,
    )


# ---------------------------------------------------------------------------
# Heuristic fallback
# ---------------------------------------------------------------------------

_CRASH_PATTERNS = re.compile(
    r"(crash|segfault|abort|panic|SIGSEGV|SIGABRT|ICE|internal compiler error"
    r"|assertion fail|stack overflow|out of memory|OOM)",
    re.IGNORECASE,
)
_REGRESSION_PATTERNS = re.compile(
    r"(regression|regress|used to work|no longer works|broke after|bisect|worked in)",
    re.IGNORECASE,
)
_FEATURE_PATTERNS = re.compile(
    r"(feature request|enhancement|RFC|proposal|would be nice|suggest)",
    re.IGNORECASE,
)
_DOCS_PATTERNS = re.compile(
    r"(documentation|typo in docs|readme|doc fix|spelling)",
    re.IGNORECASE,
)
_DUPLICATE_PATTERNS = re.compile(
    r"(duplicate of|dup of|same as #|already reported)",
    re.IGNORECASE,
)
_NEED_INFO_PATTERNS = re.compile(
    r"(need more info|cannot reproduce|unable to reproduce|please provide|steps to reproduce)",
    re.IGNORECASE,
)


def _heuristic_predict(text: str, labels: list[str] | None = None) -> Predictions:
    """Rule-based predictions from text content and GitHub labels."""
    text_lower = text.lower()
    label_str = " ".join(labels or []).lower()
    combined = text_lower + " " + label_str

    # --- is_regression ---
    is_reg = bool(_REGRESSION_PATTERNS.search(combined))
    reg_conf = 0.8 if is_reg else 0.5

    # --- issue_type ---
    issue_type = "bug"
    type_conf = 0.4
    if _REGRESSION_PATTERNS.search(combined):
        issue_type = "regression"
        type_conf = 0.7
    elif _FEATURE_PATTERNS.search(combined):
        issue_type = "feature_request"
        type_conf = 0.7
    elif _DOCS_PATTERNS.search(combined):
        issue_type = "docs"
        type_conf = 0.7
    elif "bug" in label_str or _CRASH_PATTERNS.search(combined):
        issue_type = "bug"
        type_conf = 0.6

    # --- urgency ---
    urgency = "medium"
    urg_conf = 0.4
    if _CRASH_PATTERNS.search(combined):
        urgency = "critical_bug"
        urg_conf = 0.7
    elif is_reg:
        urgency = "high"
        urg_conf = 0.6
    elif issue_type == "feature_request":
        urgency = "enhancement"
        urg_conf = 0.6
    elif issue_type == "docs":
        urgency = "low"
        urg_conf = 0.6
    elif issue_type == "question" or "question" in label_str:
        urgency = "question"
        urg_conf = 0.5

    # --- action_recommendation ---
    action = "triage"
    act_conf = 0.4
    if _DUPLICATE_PATTERNS.search(combined):
        action = "duplicate"
        act_conf = 0.7
    elif _NEED_INFO_PATTERNS.search(combined):
        action = "need_more_info"
        act_conf = 0.6
    elif urgency == "critical_bug":
        action = "assign_to_area"
        act_conf = 0.5

    return Predictions(
        urgency=urgency,
        urgency_confidence=urg_conf,
        issue_type=issue_type,
        issue_type_confidence=type_conf,
        action_recommendation=action,
        action_confidence=act_conf,
        is_regression=is_reg,
        regression_confidence=reg_conf,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict(
    text: str,
    labels: list[str] | None = None,
    embedding: list[float] | None = None,
) -> Predictions:
    """Return predictions for the given issue text.

    Args:
        text:      Issue text (title + body).
        labels:    Optional GitHub labels; used only by the heuristic fallback.
        embedding: Optional pre-computed embedding of `text`.  When provided,
                   the sklearn LR path skips re-encoding, and the SetFit path
                   routes predictions through the classification head directly,
                   so the query vector is computed only once per request.
    """
    trained = _predict_trained(text, embedding=embedding)
    if trained is not None:
        return trained
    return _heuristic_predict(text, labels=labels)


def predict_dict(
    text: str,
    labels: list[str] | None = None,
    embedding: list[float] | None = None,
) -> dict:
    """predict() but returns a plain dict for JSON serialization."""
    p = predict(text, labels=labels, embedding=embedding)
    return {
        "urgency": p.urgency,
        "urgency_confidence": round(p.urgency_confidence, 3),
        "issue_type": p.issue_type,
        "issue_type_confidence": round(p.issue_type_confidence, 3),
        "action_recommendation": p.action_recommendation,
        "action_confidence": round(p.action_confidence, 3),
        "is_regression": p.is_regression,
        "regression_confidence": round(p.regression_confidence, 3),
    }
