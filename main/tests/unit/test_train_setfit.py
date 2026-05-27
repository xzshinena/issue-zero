"""Tests for scripts/train_setfit.py helper functions and training logic."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from train_setfit import _detect_label_col, _detect_text_col, main


# ---------------------------------------------------------------------------
# Column detection helpers
# ---------------------------------------------------------------------------

class TestDetectTextCol:
    def test_finds_text_full(self):
        assert _detect_text_col({"text_full": None, "urgency": None}) == "text_full"

    def test_finds_text_fallback(self):
        assert _detect_text_col({"text": None, "label": None}) == "text"

    def test_finds_title_fallback(self):
        assert _detect_text_col({"title": None, "label": None}) == "title"

    def test_raises_on_unknown(self):
        with pytest.raises(ValueError, match="No text column"):
            _detect_text_col({"foo": None, "bar": None})


class TestDetectLabelCol:
    def test_finds_urgency(self):
        assert _detect_label_col({"text": None, "urgency": None}) == "urgency"

    def test_finds_label_fallback(self):
        assert _detect_label_col({"text": None, "label": None}) == "label"

    def test_raises_on_unknown(self):
        with pytest.raises(ValueError, match="No label column"):
            _detect_label_col({"text": None, "foo": None})


# ---------------------------------------------------------------------------
# Training entrypoint
# ---------------------------------------------------------------------------

def _mock_dataset(text_col: str = "text_full", label_col: str = "urgency"):
    split = MagicMock()
    split.features = {text_col: None, label_col: None}
    split.__len__ = lambda self: 30
    split.select.return_value = split
    split.rename_column.return_value = split
    split.map.return_value = split
    split.__getitem__ = lambda self, key: ["high"] * 30 if key == "label" else ["text"] * 30

    ds = MagicMock()
    ds.get.side_effect = lambda k, *a: split if k == "train" else None
    ds.keys.return_value = ["train"]
    ds.__getitem__ = lambda self, k: split
    return ds, split


@pytest.fixture
def fake_ml_deps(monkeypatch):
    """Inject fake datasets/setfit modules; torch is installed so only those two need faking."""
    fake_datasets = MagicMock()
    fake_setfit = MagicMock()
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    monkeypatch.setitem(sys.modules, "setfit", fake_setfit)
    return fake_datasets, fake_setfit


class TestMainEntrypoint:
    def test_labels_arg_passed_to_from_pretrained(self, monkeypatch, fake_ml_deps):
        """SetFitModel.from_pretrained must receive labels=URGENCY_LABELS — the critical pinning."""
        from app.ml.label_schema import URGENCY_LABELS
        fake_datasets, fake_setfit_mod = fake_ml_deps
        ds, _ = _mock_dataset()
        fake_datasets.load_dataset.return_value = ds
        mock_model = MagicMock()
        fake_setfit_mod.SetFitModel.from_pretrained.return_value = mock_model
        fake_setfit_mod.Trainer.return_value.evaluate.return_value = {"f1": 0.75}

        with patch("torch.backends.mps.is_available", return_value=False), \
             patch("torch.cuda.is_available", return_value=False):
            sys.argv = ["train_setfit.py"]
            main()

        _, kwargs = fake_setfit_mod.SetFitModel.from_pretrained.call_args
        assert kwargs.get("labels") == URGENCY_LABELS

    def test_column_rename_applied_for_text_full(self, monkeypatch, fake_ml_deps):
        """text_full must be renamed to 'text' before passing to Trainer."""
        fake_datasets, fake_setfit_mod = fake_ml_deps
        ds, split = _mock_dataset(text_col="text_full", label_col="urgency")
        fake_datasets.load_dataset.return_value = ds
        fake_setfit_mod.SetFitModel.from_pretrained.return_value = MagicMock()
        fake_setfit_mod.Trainer.return_value.evaluate.return_value = {}

        with patch("torch.backends.mps.is_available", return_value=False), \
             patch("torch.cuda.is_available", return_value=False):
            sys.argv = ["train_setfit.py"]
            main()

        rename_args = [c.args for c in split.rename_column.call_args_list]
        assert ("text_full", "text") in rename_args

    def test_column_rename_applied_for_urgency_label(self, monkeypatch, fake_ml_deps):
        """urgency column must be renamed to 'label' before passing to Trainer."""
        fake_datasets, fake_setfit_mod = fake_ml_deps
        ds, split = _mock_dataset(text_col="text_full", label_col="urgency")
        fake_datasets.load_dataset.return_value = ds
        fake_setfit_mod.SetFitModel.from_pretrained.return_value = MagicMock()
        fake_setfit_mod.Trainer.return_value.evaluate.return_value = {}

        with patch("torch.backends.mps.is_available", return_value=False), \
             patch("torch.cuda.is_available", return_value=False):
            sys.argv = ["train_setfit.py"]
            main()

        rename_args = [c.args for c in split.rename_column.call_args_list]
        assert ("urgency", "label") in rename_args

    def test_mps_device_selected_on_apple_silicon(self, monkeypatch, fake_ml_deps):
        fake_datasets, fake_setfit_mod = fake_ml_deps
        ds, _ = _mock_dataset()
        fake_datasets.load_dataset.return_value = ds
        mock_model = MagicMock()
        fake_setfit_mod.SetFitModel.from_pretrained.return_value = mock_model
        fake_setfit_mod.Trainer.return_value.evaluate.return_value = {}

        with patch("torch.backends.mps.is_available", return_value=True), \
             patch("torch.cuda.is_available", return_value=False):
            sys.argv = ["train_setfit.py"]
            main()

        mock_model.to.assert_called_once_with("mps")
