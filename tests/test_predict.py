"""Integration tests for scripts/predict.py — the audit accuracy entrypoint.

Mocks the fetcher and model (no real data / trained model). A "perfect" fake model
decodes a label encoded into each image, so the accuracy is deterministic and the
required stdout line can be asserted exactly. Skipped without TensorFlow.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorflow")

_PREDICT_PY = Path(__file__).resolve().parent.parent / "scripts" / "predict.py"


def _load_predict():
    spec = importlib.util.spec_from_file_location("predict_entrypoint", _PREDICT_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _LabelEncodingFetcher:
    """Fake fetcher hiding each row's label in pixel [0, 0] so a model can decode it."""

    _N = {"Training": 70, "PublicTest": 14, "PrivateTest": 14}

    def __init__(self, cfg: dict) -> None:
        pass

    def fetch(self, split: str):
        n = self._N[split]
        rng = np.random.default_rng(abs(hash(split)) % 1000)
        X = rng.integers(0, 256, (n, 48, 48), dtype=np.uint8)
        y = np.tile(np.arange(7), n // 7 + 1)[:n]
        X[:, 0, 0] = y  # encode the label into the top-left pixel
        return X, y


class _PerfectModel:
    """Decodes the label from pixel [0, 0] (÷255 rescaled) → perfect predictions."""

    def predict(self, X, **kwargs):
        X = np.asarray(X)
        labels = np.rint(X[:, 0, 0, 0] * 255).astype(int)
        onehot = np.zeros((len(X), 7), dtype=float)
        onehot[np.arange(len(X)), labels] = 1.0
        return onehot


def _cfg(tmp_path: Path) -> dict:
    return {
        "global": {"seed": 42},
        "split": {"val_size": 0.2, "stratify": True},
        "stages": {"cleaning": True, "preprocessing": True},
        "cleaning": {
            "remove_duplicates": True,
            "dedup_scope": "global",
            "drop_constant_images": True,
            "min_contrast": 0,
            "drop_non_faces": False,
            "handle_imbalance": True,
            "imbalance_strategy": "class_weight",
        },
        "preprocessing": {
            "image_size": 48,
            "grayscale": True,
            "normalization": "rescale",
            "clahe_clip_limit": 2.0,
            "clahe_tile_grid": 8,
        },
        "model": {"num_classes": 7},
        "evaluation": {"metrics": ["accuracy", "f1_macro", "confusion_matrix"]},
        "paths": {"model_save_path": str(tmp_path / "model" / "final.keras")},
    }


# ---------------------------------------------------------------------------
# score_test_set
# ---------------------------------------------------------------------------

def test_score_test_set_perfect_accuracy(tmp_path, monkeypatch) -> None:
    predict = _load_predict()
    monkeypatch.setattr(predict, "Fer2013Fetcher", _LabelEncodingFetcher)
    res = predict.score_test_set(_cfg(tmp_path), _PerfectModel())
    assert res["accuracy"] == pytest.approx(1.0)
    assert res["f1_macro"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# main() — the exact stdout contract
# ---------------------------------------------------------------------------

def test_main_prints_exact_format(tmp_path, monkeypatch, capsys) -> None:
    predict = _load_predict()
    cfg = _cfg(tmp_path)
    # a model file must exist for the load path; contents don't matter (load is mocked)
    Path(cfg["paths"]["model_save_path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["model_save_path"]).write_text("stub")

    monkeypatch.setattr(predict, "load_config", lambda _: cfg)
    monkeypatch.setattr(predict, "setup_logging", lambda _cfg: None)
    monkeypatch.setattr(predict, "Fer2013Fetcher", _LabelEncodingFetcher)
    monkeypatch.setattr(predict, "_load_keras_model", lambda _p: _PerfectModel())

    predict.main("config.yaml")

    out = capsys.readouterr().out
    assert re.search(r"^Accuracy on test set: \d+%$", out.strip(), re.MULTILINE)
    assert "Accuracy on test set: 100%" in out  # perfect model → 100%
