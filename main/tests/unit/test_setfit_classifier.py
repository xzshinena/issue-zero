"""Tests for SetFit urgency integration in classifiers.py."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import app.ml.classifiers as clf_mod
from app.ml.label_schema import URGENCY_LABELS, ISSUE_TYPE_LABELS


@pytest.fixture(autouse=True)
def reset_classifier_state():
    clf_mod._setfit_models = {}
    clf_mod._setfit_loaded = False
    clf_mod._trained_models = {}
    clf_mod._models_loaded = False
    yield
    clf_mod._setfit_models = {}
    clf_mod._setfit_loaded = False
    clf_mod._trained_models = {}
    clf_mod._models_loaded = False


@pytest.fixture
def fake_setfit_mod(monkeypatch):
    """Inject a fake setfit module so tests run without setfit installed."""
    fake = MagicMock()
    monkeypatch.setitem(sys.modules, "setfit", fake)
    return fake


def _setfit_mock(label: str = "high", labels: list[str] = URGENCY_LABELS) -> MagicMock:
    proba = np.zeros(len(labels))
    proba[labels.index(label)] = 0.9
    m = MagicMock()
    m.predict.return_value = [label]
    m.predict_proba.return_value = [proba]
    # model_head for direct embedding path
    head = MagicMock()
    head.predict_proba.return_value = [proba]
    m.model_head = head
    return m


def _lr_mock(label: str = "bug", conf: float = 0.8) -> MagicMock:
    m = MagicMock()
    m.predict.return_value = [label]
    m.predict_proba.return_value = [[conf, 1.0 - conf]]
    # expose clf/le for the direct-embedding path
    le = MagicMock()
    le.inverse_transform.return_value = [label]
    clf = MagicMock()
    clf.predict.return_value = [0]
    clf.predict_proba.return_value = [[conf, 1.0 - conf]]
    m.clf = clf
    m.le = le
    return m


def _patch_all_lr_tasks():
    clf_mod._trained_models = {
        "urgency": _lr_mock("medium"),
        "issue_type": _lr_mock("bug"),
        "action_recommendation": _lr_mock("triage"),
        "is_regression": _lr_mock("false"),
    }
    clf_mod._models_loaded = True


# ---------------------------------------------------------------------------
# _try_load_setfit — loads per-task
# ---------------------------------------------------------------------------

class TestTryLoadSetfit:
    def test_returns_true_and_sets_model(self, tmp_path, monkeypatch, fake_setfit_mod):
        (tmp_path / "urgency-setfit").mkdir()
        monkeypatch.setattr(clf_mod, "MODELS_DIR", tmp_path)
        mock_model = _setfit_mock()
        fake_setfit_mod.SetFitModel.from_pretrained.return_value = mock_model
        result = clf_mod._try_load_setfit()
        assert result is True
        assert clf_mod._setfit_models.get("urgency") is mock_model

    def test_loads_multiple_tasks(self, tmp_path, monkeypatch, fake_setfit_mod):
        (tmp_path / "urgency-setfit").mkdir()
        (tmp_path / "issue_type-setfit").mkdir()
        monkeypatch.setattr(clf_mod, "MODELS_DIR", tmp_path)
        fake_setfit_mod.SetFitModel.from_pretrained.return_value = _setfit_mock()
        clf_mod._try_load_setfit()
        assert "urgency" in clf_mod._setfit_models
        assert "issue_type" in clf_mod._setfit_models

    def test_returns_false_when_no_dirs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(clf_mod, "MODELS_DIR", tmp_path)
        assert clf_mod._try_load_setfit() is False

    def test_skips_task_on_load_exception(self, tmp_path, monkeypatch, fake_setfit_mod):
        (tmp_path / "urgency-setfit").mkdir()
        monkeypatch.setattr(clf_mod, "MODELS_DIR", tmp_path)
        fake_setfit_mod.SetFitModel.from_pretrained.side_effect = RuntimeError("corrupt")
        result = clf_mod._try_load_setfit()
        assert result is False
        assert "urgency" not in clf_mod._setfit_models

    def test_caches_after_first_call(self, tmp_path, monkeypatch):
        monkeypatch.setattr(clf_mod, "MODELS_DIR", tmp_path)
        clf_mod._try_load_setfit()
        clf_mod._try_load_setfit()
        assert clf_mod._setfit_loaded is True


# ---------------------------------------------------------------------------
# Per-task independence
# ---------------------------------------------------------------------------

class TestPerTaskIndependence:
    def test_setfit_urgency_with_lr_issue_type(self, tmp_path, monkeypatch, fake_setfit_mod):
        """SetFit handles urgency; LR handles issue_type — both present simultaneously."""
        (tmp_path / "urgency-setfit").mkdir()
        monkeypatch.setattr(clf_mod, "MODELS_DIR", tmp_path)
        fake_setfit_mod.SetFitModel.from_pretrained.return_value = _setfit_mock("critical_bug")
        clf_mod._trained_models = {
            "issue_type": _lr_mock("regression"),
            "action_recommendation": _lr_mock("triage"),
            "is_regression": _lr_mock("true"),
        }
        clf_mod._models_loaded = True

        result = clf_mod.predict("segfault on startup")
        assert result.urgency == "critical_bug"
        assert result.issue_type == "regression"

    def test_heuristic_fills_missing_task(self, tmp_path, monkeypatch, fake_setfit_mod):
        """If only urgency SetFit is loaded and no LR for others, heuristic fills the rest."""
        (tmp_path / "urgency-setfit").mkdir()
        monkeypatch.setattr(clf_mod, "MODELS_DIR", tmp_path)
        fake_setfit_mod.SetFitModel.from_pretrained.return_value = _setfit_mock("high")
        clf_mod._models_loaded = True  # no sklearn models

        result = clf_mod.predict("memory leak regression")
        assert result.urgency == "high"
        # heuristic should have produced something for the rest
        assert isinstance(result.issue_type, str)
        assert isinstance(result.action_recommendation, str)


# ---------------------------------------------------------------------------
# Pre-computed embedding path (dedup)
# ---------------------------------------------------------------------------

class TestEmbeddingDedup:
    def test_lr_skips_reencoding_when_embedding_provided(self, tmp_path, monkeypatch):
        """sklearn LR should call clf.predict() directly when embedding is supplied."""
        monkeypatch.setattr(clf_mod, "MODELS_DIR", tmp_path)
        _patch_all_lr_tasks()

        embedding = [0.1] * 384
        with patch("app.ml.train.feature_extractor.extract_features") as mock_extract:
            clf_mod.predict("segfault", embedding=embedding)

        mock_extract.assert_not_called()

    def test_setfit_head_called_when_embedding_provided(self, tmp_path, monkeypatch, fake_setfit_mod):
        """SetFit model_head.predict_proba() should be called when embedding is supplied."""
        (tmp_path / "urgency-setfit").mkdir()
        monkeypatch.setattr(clf_mod, "MODELS_DIR", tmp_path)
        mock_model = _setfit_mock("high")
        fake_setfit_mod.SetFitModel.from_pretrained.return_value = mock_model

        clf_mod._trained_models = {
            "issue_type": _lr_mock("bug"),
            "action_recommendation": _lr_mock("triage"),
            "is_regression": _lr_mock("false"),
        }
        clf_mod._models_loaded = True

        embedding = [0.1] * 384
        result = clf_mod.predict("crash on startup", embedding=embedding)

        mock_model.model_head.predict_proba.assert_called_once()
        assert result.urgency == "high"

    def test_predict_dict_accepts_embedding(self, tmp_path, monkeypatch):
        """predict_dict() should pass embedding through to predict()."""
        monkeypatch.setattr(clf_mod, "MODELS_DIR", tmp_path)
        clf_mod._models_loaded = True  # no models, will use heuristic

        embedding = [0.0] * 384
        result = clf_mod.predict_dict("crash", embedding=embedding)
        assert "urgency" in result


# ---------------------------------------------------------------------------
# Label order pinning
# ---------------------------------------------------------------------------

class TestLabelOrderPinning:
    def test_predict_proba_confidence_maps_to_correct_class(self, tmp_path, monkeypatch, fake_setfit_mod):
        """Confidence must reflect the predicted label, not an arbitrary column."""
        (tmp_path / "urgency-setfit").mkdir()
        monkeypatch.setattr(clf_mod, "MODELS_DIR", tmp_path)
        _patch_all_lr_tasks()

        target = "medium"
        proba = np.zeros(len(URGENCY_LABELS))
        proba[URGENCY_LABELS.index(target)] = 0.77
        mock_model = MagicMock()
        mock_model.predict.return_value = [target]
        mock_model.predict_proba.return_value = [proba]
        head = MagicMock()
        head.predict_proba.return_value = [proba]
        mock_model.model_head = head
        fake_setfit_mod.SetFitModel.from_pretrained.return_value = mock_model

        result = clf_mod.predict("minor formatting issue")

        assert result.urgency == "medium"
        assert abs(result.urgency_confidence - 0.77) < 0.01


# ---------------------------------------------------------------------------
# Regression: full heuristic fallback
# ---------------------------------------------------------------------------

class TestHeuristicFallback:
    def test_falls_back_to_heuristic_when_no_models(self, tmp_path, monkeypatch):
        monkeypatch.setattr(clf_mod, "MODELS_DIR", tmp_path)
        clf_mod._models_loaded = True
        result = clf_mod.predict("crash SIGSEGV on startup")
        assert result.urgency == "critical_bug"

    def test_heuristic_used_when_all_model_tasks_fail(self, tmp_path, monkeypatch, fake_setfit_mod):
        """If all per-task model predictions raise, heuristic fills all tasks."""
        (tmp_path / "urgency-setfit").mkdir()
        monkeypatch.setattr(clf_mod, "MODELS_DIR", tmp_path)
        fake_setfit_mod.SetFitModel.from_pretrained.return_value = _setfit_mock("high")
        # Sabotage the SetFit model so every call raises
        clf_mod._setfit_models["urgency"] = MagicMock(
            predict=MagicMock(side_effect=RuntimeError("fail")),
            model_head=MagicMock(predict_proba=MagicMock(side_effect=RuntimeError("fail"))),
        )
        clf_mod._models_loaded = True

        result = clf_mod.predict("crash")
        assert isinstance(result.urgency, str)
