"""
Classifier loading and inference for:
  - urgency        (critical_bug, high, medium, low, enhancement, question)
  - issue_type     (bug, feature_request, docs, refactor, regression, question)
  - action_recommendation (triage, assign_to_area, need_more_info, duplicate, close)
  - is_regression  (True / False)

Two modes:
  1. Trained scikit-learn models (joblib artifacts in models/ dir) — preferred.
  2. Heuristic fallback using label keywords + text patterns.

At startup, try to load trained models; fall back to heuristics if not found.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.ml.label_schema import ACTION_LABELS, ISSUE_TYPE_LABELS, URGENCY_LABELS

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"


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
# Trained model loader
# ---------------------------------------------------------------------------

_trained_models: dict[str, Any] = {}
_models_loaded = False


def _try_load_trained() -> bool:
    """Attempt to load joblib models from MODELS_DIR. Returns True if all four found."""
    global _trained_models, _models_loaded
    if _models_loaded:
        return bool(_trained_models)

    _models_loaded = True
    names = ["urgency", "issue_type", "action_recommendation", "is_regression"]
    try:
        import joblib
    except ImportError:
        return False

    all_found = True
    for name in names:
        path = MODELS_DIR / f"{name}.joblib"
        if path.exists():
            _trained_models[name] = joblib.load(path)
        else:
            all_found = False
    return all_found


def _predict_trained(text: str) -> Predictions | None:
    """Run sklearn models if they are loaded. Returns None if not available."""
    if not _trained_models:
        if not _try_load_trained():
            return None
    if not _trained_models:
        return None

    results: dict[str, tuple[str, float]] = {}
    for name in ["urgency", "issue_type", "action_recommendation", "is_regression"]:
        model = _trained_models.get(name)
        if model is None:
            return None
        pred = model.predict([text])[0]
        conf = 0.0
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([text])[0]
            conf = float(max(proba))
        elif hasattr(model, "decision_function"):
            conf = 0.7
        results[name] = (str(pred), conf)

    return Predictions(
        urgency=results["urgency"][0],
        urgency_confidence=results["urgency"][1],
        issue_type=results["issue_type"][0],
        issue_type_confidence=results["issue_type"][1],
        action_recommendation=results["action_recommendation"][0],
        action_confidence=results["action_recommendation"][1],
        is_regression=results["is_regression"][0].lower() in ("true", "1", "yes"),
        regression_confidence=results["is_regression"][1],
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

def predict(text: str, labels: list[str] | None = None) -> Predictions:
    """
    Return predictions for the given issue text.
    Uses trained models if available, else heuristic fallback.
    """
    trained = _predict_trained(text)
    if trained is not None:
        return trained
    return _heuristic_predict(text, labels=labels)


def predict_dict(text: str, labels: list[str] | None = None) -> dict:
    """predict() but returns a plain dict for JSON serialization."""
    p = predict(text, labels=labels)
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
