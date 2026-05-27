"""Versioned model registry.

Directory layout::

    models/
      <task>/
        v1/
          model.joblib       # sklearn LabeledClassifier
          metadata.json
        v2/
          setfit/            # SetFit model directory
          metadata.json

metadata.json fields:
  task, version, model_type ("sklearn" | "setfit"), created_at, extra (dict)

Flat legacy layout (models/<task>.joblib, models/<task>-setfit/) is still
read by classifiers.py as a fallback when no versioned models exist.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

_METADATA_FILE = "metadata.json"
_JOBLIB_FILE = "model.joblib"
_SETFIT_DIR = "setfit"


# ---------------------------------------------------------------------------
# Version enumeration
# ---------------------------------------------------------------------------

def list_versions(task: str, models_dir: Path | None = None) -> list[int]:
    """Return all available version numbers for a task, sorted ascending."""
    root = (models_dir or MODELS_DIR) / task
    if not root.is_dir():
        return []
    versions = []
    for child in root.iterdir():
        if child.is_dir() and child.name.startswith("v"):
            try:
                versions.append(int(child.name[1:]))
            except ValueError:
                pass
    return sorted(versions)


def latest_version(task: str, models_dir: Path | None = None) -> int | None:
    """Return the highest version number for a task, or None."""
    vs = list_versions(task, models_dir)
    return vs[-1] if vs else None


def version_path(task: str, version: int, models_dir: Path | None = None) -> Path:
    return (models_dir or MODELS_DIR) / task / f"v{version}"


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_model(
    task: str,
    model: Any,
    model_type: str,
    extra: dict | None = None,
    models_dir: Path | None = None,
) -> int:
    """Save model as the next version. Returns the new version number.

    Args:
        task:       Task name (e.g. "urgency").
        model:      sklearn LabeledClassifier OR path to a trained SetFit dir.
        model_type: "sklearn" or "setfit".
        extra:      Optional dict of additional metadata to record.
        models_dir: Override base models dir (for testing).
    """
    root = models_dir or MODELS_DIR
    current = latest_version(task, root) or 0
    new_version = current + 1
    vdir = root / task / f"v{new_version}"
    vdir.mkdir(parents=True, exist_ok=True)

    if model_type == "sklearn":
        import joblib  # noqa: PLC0415
        joblib.dump(model, vdir / _JOBLIB_FILE)
    elif model_type == "setfit":
        # `model` is expected to be a SetFit model instance
        model.save_pretrained(str(vdir / _SETFIT_DIR))
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}. Use 'sklearn' or 'setfit'.")

    metadata = {
        "task": task,
        "version": new_version,
        "model_type": model_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "extra": extra or {},
    }
    (vdir / _METADATA_FILE).write_text(json.dumps(metadata, indent=2))
    return new_version


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_model(
    task: str,
    version: int | None = None,
    models_dir: Path | None = None,
) -> tuple[Any, dict] | None:
    """Load a versioned model. Returns (model, metadata) or None.

    When version is None, the latest version is loaded.
    """
    root = models_dir or MODELS_DIR
    ver = version if version is not None else latest_version(task, root)
    if ver is None:
        return None

    vdir = root / task / f"v{ver}"
    meta_path = vdir / _METADATA_FILE
    if not meta_path.exists():
        return None

    metadata = json.loads(meta_path.read_text())
    model_type = metadata.get("model_type", "sklearn")

    try:
        if model_type == "sklearn":
            import joblib  # noqa: PLC0415
            model = joblib.load(vdir / _JOBLIB_FILE)
        elif model_type == "setfit":
            from setfit import SetFitModel  # noqa: PLC0415
            model = SetFitModel.from_pretrained(str(vdir / _SETFIT_DIR))
        else:
            return None
    except Exception:
        return None

    return model, metadata


# ---------------------------------------------------------------------------
# Convenience: load latest of every task
# ---------------------------------------------------------------------------

def load_all_latest(
    model_type: str | None = None,
    models_dir: Path | None = None,
) -> dict[str, tuple[Any, dict]]:
    """Load the latest versioned model for each task.

    Args:
        model_type: If set ("sklearn" or "setfit"), only load models of that type.
    """
    root = models_dir or MODELS_DIR
    if not root.is_dir():
        return {}

    result: dict[str, tuple[Any, dict]] = {}
    for task_dir in root.iterdir():
        if not task_dir.is_dir():
            continue
        task = task_dir.name
        # Skip flat legacy dirs (e.g. "urgency-setfit")
        if "-" in task:
            continue
        loaded = load_model(task, models_dir=root)
        if loaded is None:
            continue
        model, meta = loaded
        if model_type and meta.get("model_type") != model_type:
            continue
        result[task] = (model, meta)

    return result
