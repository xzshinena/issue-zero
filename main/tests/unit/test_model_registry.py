"""Unit tests for app.ml.model_registry."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture()
def tmp_registry(tmp_path):
    return tmp_path


class TestVersionListing:
    def test_empty_returns_empty_list(self, tmp_registry):
        from app.ml.model_registry import list_versions
        assert list_versions("urgency", tmp_registry) == []

    def test_latest_none_when_empty(self, tmp_registry):
        from app.ml.model_registry import latest_version
        assert latest_version("urgency", tmp_registry) is None

    def test_lists_versions_sorted(self, tmp_registry):
        from app.ml.model_registry import list_versions
        for v in (3, 1, 2):
            (tmp_registry / "urgency" / f"v{v}").mkdir(parents=True)
        assert list_versions("urgency", tmp_registry) == [1, 2, 3]

    def test_latest_returns_highest(self, tmp_registry):
        from app.ml.model_registry import latest_version
        for v in (1, 2, 5):
            (tmp_registry / "urgency" / f"v{v}").mkdir(parents=True)
        assert latest_version("urgency", tmp_registry) == 5

    def test_ignores_non_version_dirs(self, tmp_registry):
        from app.ml.model_registry import list_versions
        (tmp_registry / "urgency" / "v1").mkdir(parents=True)
        (tmp_registry / "urgency" / "old").mkdir(parents=True)
        assert list_versions("urgency", tmp_registry) == [1]


def _patch_joblib(monkeypatch):
    """Inject a fake joblib module so tests don't need real serializable objects."""
    import sys
    _store: dict[str, object] = {}

    fake_jl = MagicMock()
    fake_jl.dump.side_effect = lambda obj, path, **kw: _store.__setitem__(str(path), obj)
    fake_jl.load.side_effect = lambda path: _store[str(path)]

    monkeypatch.setitem(sys.modules, "joblib", fake_jl)
    return _store


class TestSaveAndLoad:
    def test_save_sklearn_increments_version(self, tmp_registry, monkeypatch):
        from app.ml.model_registry import save_model, list_versions
        _patch_joblib(monkeypatch)

        model = MagicMock()
        v1 = save_model("urgency", model, "sklearn", models_dir=tmp_registry)
        v2 = save_model("urgency", model, "sklearn", models_dir=tmp_registry)
        assert v1 == 1
        assert v2 == 2
        assert list_versions("urgency", tmp_registry) == [1, 2]

    def test_save_writes_metadata(self, tmp_registry, monkeypatch):
        from app.ml.model_registry import save_model, version_path
        _patch_joblib(monkeypatch)

        model = MagicMock()
        save_model("urgency", model, "sklearn", extra={"n_examples": 42}, models_dir=tmp_registry)
        meta = json.loads((version_path("urgency", 1, tmp_registry) / "metadata.json").read_text())
        assert meta["task"] == "urgency"
        assert meta["version"] == 1
        assert meta["model_type"] == "sklearn"
        assert meta["extra"]["n_examples"] == 42

    def test_load_returns_none_when_no_model(self, tmp_registry):
        from app.ml.model_registry import load_model
        assert load_model("urgency", models_dir=tmp_registry) is None

    def test_load_latest_after_save(self, tmp_registry, monkeypatch):
        from app.ml.model_registry import save_model, load_model
        _patch_joblib(monkeypatch)

        fake_clf = object()
        save_model("urgency", fake_clf, "sklearn", models_dir=tmp_registry)
        result = load_model("urgency", models_dir=tmp_registry)
        assert result is not None
        model, meta = result
        assert meta["version"] == 1
        assert meta["model_type"] == "sklearn"

    def test_load_specific_version(self, tmp_registry, monkeypatch):
        from app.ml.model_registry import save_model, load_model
        _patch_joblib(monkeypatch)

        save_model("urgency", object(), "sklearn", models_dir=tmp_registry)
        save_model("urgency", object(), "sklearn", models_dir=tmp_registry)

        result = load_model("urgency", version=1, models_dir=tmp_registry)
        assert result is not None
        _, meta = result
        assert meta["version"] == 1

    def test_save_unknown_model_type_raises(self, tmp_registry):
        from app.ml.model_registry import save_model
        with pytest.raises(ValueError, match="Unknown model_type"):
            save_model("urgency", MagicMock(), "xgboost", models_dir=tmp_registry)


class TestLoadAllLatest:
    def test_loads_multiple_tasks(self, tmp_registry, monkeypatch):
        from app.ml.model_registry import save_model, load_all_latest
        _patch_joblib(monkeypatch)

        for task in ("urgency", "issue_type"):
            save_model(task, object(), "sklearn", models_dir=tmp_registry)

        loaded = load_all_latest(model_type="sklearn", models_dir=tmp_registry)
        assert set(loaded.keys()) == {"urgency", "issue_type"}

    def test_filters_by_model_type(self, tmp_registry, monkeypatch):
        from app.ml.model_registry import save_model, load_all_latest
        _patch_joblib(monkeypatch)

        save_model("urgency", object(), "sklearn", models_dir=tmp_registry)
        loaded = load_all_latest(model_type="setfit", models_dir=tmp_registry)
        assert "urgency" not in loaded

    def test_returns_empty_when_no_models(self, tmp_registry):
        from app.ml.model_registry import load_all_latest
        assert load_all_latest(models_dir=tmp_registry) == {}
