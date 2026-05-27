"""Tests for LabeledClassifier and train_all."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pytest


_FAKE_DIM = 4
_FAKE_VEC = [[0.1, 0.2, 0.3, 0.4]]


def _fake_extract(texts):
    return np.array([_FAKE_VEC[0]] * len(texts), dtype=np.float32)


@pytest.fixture
def simple_classifier():
    from app.ml.train.trainer import train_task

    texts = ["crash on startup"] * 6 + ["new feature request"] * 6
    labels = ["critical_bug"] * 6 + ["enhancement"] * 6
    with patch("app.ml.train.trainer.extract_features", side_effect=_fake_extract):
        return train_task("urgency", texts, labels)


def test_labeled_classifier_predict(simple_classifier):
    with patch("app.ml.train.feature_extractor.embed_batch", return_value=_FAKE_VEC):
        result = simple_classifier.predict(["crash on startup"])
    assert isinstance(result, list)
    assert isinstance(result[0], str)


def test_labeled_classifier_predict_proba(simple_classifier):
    with patch("app.ml.train.feature_extractor.embed_batch", return_value=_FAKE_VEC):
        proba = simple_classifier.predict_proba(["crash on startup"])
    assert proba.ndim == 2
    assert proba.shape[0] == 1


def test_labeled_classifier_picklable(simple_classifier, tmp_path):
    path = tmp_path / "model.joblib"
    joblib.dump(simple_classifier, path)
    loaded = joblib.load(path)
    assert loaded.task_name == simple_classifier.task_name
    with patch("app.ml.train.feature_extractor.embed_batch", return_value=_FAKE_VEC):
        assert loaded.predict(["test"]) is not None


def test_train_task_returns_labeled_classifier():
    from app.ml.train.trainer import LabeledClassifier, train_task

    texts = ["bug"] * 4 + ["feature"] * 4
    labels = ["bug"] * 4 + ["feature_request"] * 4
    with patch("app.ml.train.trainer.extract_features", side_effect=_fake_extract):
        model = train_task("issue_type", texts, labels)
    assert isinstance(model, LabeledClassifier)


def test_train_all_creates_versioned_dirs(tmp_path, sample_issues_jsonl):
    from app.ml.train.data_loader import load_records
    from app.ml.train.trainer import train_all

    records = load_records(sample_issues_jsonl)
    with patch("app.ml.train.trainer.extract_features", side_effect=_fake_extract):
        train_all(records, models_dir=tmp_path)

    for task in ("urgency", "issue_type", "action_recommendation", "is_regression"):
        assert (tmp_path / task / "v1" / "model.joblib").exists(), f"missing {task}/v1/model.joblib"
        assert (tmp_path / task / "v1" / "metadata.json").exists(), f"missing {task}/v1/metadata.json"


def test_train_all_flat_layout(tmp_path, sample_issues_jsonl):
    from app.ml.train.data_loader import load_records
    from app.ml.train.trainer import train_all

    records = load_records(sample_issues_jsonl)
    with patch("app.ml.train.trainer.extract_features", side_effect=_fake_extract):
        train_all(records, models_dir=tmp_path, use_registry=False)

    created = list(tmp_path.glob("*.joblib"))
    assert len(created) == 4
    names = {p.stem for p in created}
    assert names == {"urgency", "issue_type", "action_recommendation", "is_regression"}
