"""Tests for SetFit urgency integration in classifiers.py."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

import app.ml.classifiers as clf_mod
from app.ml.label_schema import URGENCY_LABELS


@pytest.fixture(autouse=True)
def reset_classifier_state():
    clf_mod._setfit_model = None
    clf_mod._setfit_loaded = False
    clf_mod._trained_models = {}
    clf_mod._models_loaded = False
    yield
    clf_mod._setfit_model = None
    clf_mod._setfit_loaded = False
    clf_mod._trained_models = {}
    clf_mod._models_loaded = False


@pytest.fixture
def fake_setfit_mod(monkeypatch):
    """Inject a fake setfit module so tests run without setfit installed."""
    fake = MagicMock()
    monkeypatch.setitem(sys.modules, "setfit", fake)
    return fake


def _setfit_mock(label: str = "high") -> MagicMock:
    proba = np.zeros(len(URGENCY_LABELS))
    proba[URGENCY_LABELS.index(label)] = 0.9
    m = MagicMock()
    m.predict.return_value = [label]
    m.predict_proba.return_value = [proba]
    return m


def _lr_mock(label: str = "bug", conf: float = 0.8) -> MagicMock:
    m = MagicMock()
    m.predict.return_value = [label]
    m.predict_proba.return_value = [[conf, 1.0 - conf]]
    return m


def _patch_three_lr_tasks():
    clf_mod._trained_models = {
        "issue_type": _lr_mock("bug"),
        "action_recommendation": _lr_mock("triage"),
        "is_regression": _lr_mock("false"),
    }
    clf_mod._models_loaded = True


# ---------------------------------------------------------------------------
# _try_load_setfit
# ---------------------------------------------------------------------------

class TestTryLoadSetfit:
    def test_returns_true_and_sets_model(self, tmp_path, monkeypatch, fake_setfit_mod):
        (tmp_path / "urgency-setfit").mkdir()
        monkeypatch.setattr(clf_mod, "MODELS_DIR", tmp_path)
        mock_model = _setfit_mock()
        fake_setfit_mod.SetFitModel.from_pretrained.return_value = mock_model
        result = clf_mod._try_load_setfit()
        assert result is True
        assert clf_mod._setfit_model is mock_model

    def test_returns_false_when_dir_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(clf_mod, "MODELS_DIR", tmp_path)
        assert clf_mod._try_load_setfit() is False
        assert clf_mod._setfit_model is None

    def test_returns_false_on_load_exception(self, tmp_path, monkeypatch, fake_setfit_mod):
        (tmp_path / "urgency-setfit").mkdir()
        monkeypatch.setattr(clf_mod, "MODELS_DIR", tmp_path)
        fake_setfit_mod.SetFitModel.from_pretrained.side_effect = RuntimeError("corrupt model")
        result = clf_mod._try_load_setfit()
        assert result is False
        assert clf_mod._setfit_model is None

    def test_caches_after_first_call(self, tmp_path, monkeypatch):
        monkeypatch.setattr(clf_mod, "MODELS_DIR", tmp_path)
        clf_mod._try_load_setfit()
        clf_mod._try_load_setfit()  # second call — must not re-attempt
        assert clf_mod._setfit_loaded is True


# ---------------------------------------------------------------------------
# Urgency source priority: SetFit > LR
# ---------------------------------------------------------------------------

class TestUrgencySourcePriority:
    def test_setfit_wins_over_lr_urgency(self, tmp_path, monkeypatch, fake_setfit_mod):
        (tmp_path / "urgency-setfit").mkdir()
        monkeypatch.setattr(clf_mod, "MODELS_DIR", tmp_path)
        _patch_three_lr_tasks()
        clf_mod._trained_models["urgency"] = _lr_mock("low")  # LR says "low"
        fake_setfit_mod.SetFitModel.from_pretrained.return_value = _setfit_mock("critical_bug")
        result = clf_mod.predict("segfault on startup")
        assert result.urgency == "critical_bug"  # SetFit wins

    def test_falls_back_to_lr_urgency_when_no_setfit(self, tmp_path, monkeypatch):
        monkeypatch.setattr(clf_mod, "MODELS_DIR", tmp_path)  # no urgency-setfit/
        _patch_three_lr_tasks()
        clf_mod._trained_models["urgency"] = _lr_mock("enhancement")

        result = clf_mod.predict("add dark mode")
        assert result.urgency == "enhancement"

    def test_falls_back_to_heuristic_when_no_models(self, tmp_path, monkeypatch):
        monkeypatch.setattr(clf_mod, "MODELS_DIR", tmp_path)
        clf_mod._models_loaded = True  # pretend load was tried, nothing found
        result = clf_mod.predict("crash SIGSEGV on startup")
        assert result.urgency == "critical_bug"  # heuristic catches crash pattern


# ---------------------------------------------------------------------------
# Label order — the critical gap
# ---------------------------------------------------------------------------

class TestLabelOrderPinning:
    def test_predict_proba_confidence_maps_to_correct_class(self, tmp_path, monkeypatch, fake_setfit_mod):
        """Confidence must reflect the predicted label, not an arbitrary column."""
        (tmp_path / "urgency-setfit").mkdir()
        monkeypatch.setattr(clf_mod, "MODELS_DIR", tmp_path)
        _patch_three_lr_tasks()

        target = "medium"
        proba = np.zeros(len(URGENCY_LABELS))
        proba[URGENCY_LABELS.index(target)] = 0.77
        mock_model = MagicMock()
        mock_model.predict.return_value = [target]
        mock_model.predict_proba.return_value = [proba]
        fake_setfit_mod.SetFitModel.from_pretrained.return_value = mock_model

        result = clf_mod.predict("minor formatting issue")

        assert result.urgency == "medium"
        assert abs(result.urgency_confidence - 0.77) < 0.01


# ---------------------------------------------------------------------------
# Regression: remaining three tasks still served by LR after restructure
# ---------------------------------------------------------------------------

class TestLRTasksAfterRestructure:
    def test_issue_type_served_by_lr(self, tmp_path, monkeypatch, fake_setfit_mod):
        (tmp_path / "urgency-setfit").mkdir()
        monkeypatch.setattr(clf_mod, "MODELS_DIR", tmp_path)
        _patch_three_lr_tasks()
        clf_mod._trained_models["issue_type"] = _lr_mock("feature_request")
        fake_setfit_mod.SetFitModel.from_pretrained.return_value = _setfit_mock("medium")
        result = clf_mod.predict("add dark mode support")
        assert result.issue_type == "feature_request"

    def test_returns_none_when_lr_tasks_missing(self, tmp_path, monkeypatch, fake_setfit_mod):
        """If LR models for issue_type/action/regression are absent, fall to heuristic."""
        (tmp_path / "urgency-setfit").mkdir()
        monkeypatch.setattr(clf_mod, "MODELS_DIR", tmp_path)
        clf_mod._models_loaded = True  # no LR models loaded
        fake_setfit_mod.SetFitModel.from_pretrained.return_value = _setfit_mock("high")
        result = clf_mod.predict("null pointer exception")

        # _predict_trained returns None → heuristic kicks in
        assert isinstance(result.issue_type, str)
        assert isinstance(result.urgency, str)
