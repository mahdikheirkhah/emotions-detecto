"""Integration test for scripts/train.py — the whole pipeline on tiny fake data.

Mocks the dataset fetcher (no real FER-2013 needed) and runs one epoch end-to-end,
asserting the orchestration wires together and persists its outputs. Skipped where
TensorFlow is unavailable.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorflow")

_TRAIN_PY = Path(__file__).resolve().parent.parent / "scripts" / "train.py"


def _load_train_module():
    spec = importlib.util.spec_from_file_location("train_entrypoint", _TRAIN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeFetcher:
    """Returns tiny, class-balanced fake splits instead of reading CSVs."""

    _N = {"Training": 70, "PublicTest": 14, "PrivateTest": 14}

    def __init__(self, cfg: dict) -> None:
        pass

    def fetch(self, split: str):
        n = self._N[split]
        rng = np.random.default_rng(abs(hash(split)) % 1000)
        X = rng.integers(0, 256, (n, 48, 48), dtype=np.uint8)
        y = np.tile(np.arange(7), n // 7 + 1)[:n]  # all 7 classes present
        return X, y


def _cfg(tmp_path: Path, **stage_over) -> dict:
    model_dir = tmp_path / "model"
    stages = {
        "cleaning": True,
        "preprocessing": True,
        "augmentation": True,
        "decomposition": False,
        "tuning": False,
    }
    stages.update(stage_over)
    return {
        "global": {"seed": 42, "log_level": "INFO"},
        "stages": stages,
        "split": {"val_size": 0.2, "stratify": True},
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
        "augmentation": {
            "strategy": "basic",
            "horizontal_flip": True,
            "rotation_range": 10,
            "zoom_range": 0.1,
            "width_shift_range": 0.1,
            "height_shift_range": 0.1,
        },
        "model": {
            "architecture": "simple_cnn",
            "optimizer": "adam",
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 1,
            "dropout_rate": 0.5,
            "num_conv_blocks": 3,
            "filters_start": 32,
            "kernel_size": 3,
            "convs_per_block": 2,
            "loss": "categorical_crossentropy",
            "num_classes": 7,
        },
        "pipeline": {"shuffle_buffer": 100, "cache": True},
        "callbacks": {
            "monitor": "val_loss",
            "early_stopping": True,
            "tensorboard": False,  # avoid the optional tensorboard package dep
            "early_stopping_patience": 10,
            "reduce_lr_patience": 5,
            "reduce_lr_factor": 0.5,
            "min_lr": 1e-6,
        },
        "paths": {
            "model_save_path": str(model_dir / "final.keras"),
            "arch_txt_path": str(model_dir / "arch.txt"),
            "tensorboard_dir": str(tmp_path / "tb"),
            "results_dir": str(tmp_path),
            "logs_dir": str(tmp_path / "logs"),
        },
    }


def _run(tmp_path, monkeypatch, **stage_over):
    train = _load_train_module()
    monkeypatch.setattr(train, "Fer2013Fetcher", _FakeFetcher)
    return train, train.run(_cfg(tmp_path, **stage_over))


# ---------------------------------------------------------------------------
# end-to-end orchestration
# ---------------------------------------------------------------------------


def test_run_trains_and_returns_history(tmp_path, monkeypatch) -> None:
    _, history = _run(tmp_path, monkeypatch)
    assert "loss" in history.history
    assert "val_loss" in history.history
    assert "val_accuracy" in history.history


def test_run_saves_model(tmp_path, monkeypatch) -> None:
    _run(tmp_path, monkeypatch)
    assert (tmp_path / "model" / "final.keras").exists()


def test_run_saves_history_json(tmp_path, monkeypatch) -> None:
    _run(tmp_path, monkeypatch)
    hist_path = tmp_path / "model" / "history.json"
    assert hist_path.exists()
    data = json.loads(hist_path.read_text())
    assert "val_loss" in data and isinstance(data["val_loss"], list)


def test_run_writes_architecture_summary(tmp_path, monkeypatch) -> None:
    _run(tmp_path, monkeypatch)
    arch = (tmp_path / "model" / "arch.txt").read_text()
    assert "Total params" in arch


def test_run_stage_off_still_trains(tmp_path, monkeypatch) -> None:
    # a raw-pixel, no-clean, no-aug ablation must still run end-to-end
    _, history = _run(
        tmp_path, monkeypatch, cleaning=False, preprocessing=False, augmentation=False
    )
    assert "val_loss" in history.history


# ---------------------------------------------------------------------------
# stages.tuning wiring (search runs BEFORE the final train)
# ---------------------------------------------------------------------------


class _FakeTuner:
    """Stands in for a keras_tuner tuner — search is a no-op."""

    def search(self, *args, **kwargs) -> None:
        pass


def test_run_tuning_applies_best_hyperparameters(tmp_path, monkeypatch) -> None:
    # _run_tuning must splice the winning values into the config it returns.
    train = _load_train_module()
    import src.emotion_detector.models.tuning as tmod

    monkeypatch.setattr(tmod, "make_tuner", lambda cfg: _FakeTuner())
    monkeypatch.setattr(
        tmod,
        "best_hyperparameters",
        lambda tuner: {"learning_rate": 0.01, "optimizer": "sgd"},
    )
    monkeypatch.setattr(tmod, "results_table", lambda tuner: None)
    monkeypatch.setattr(tmod, "save_results_table", lambda df, cfg: {})

    cfg = _cfg(tmp_path)
    cfg["tuning"] = {"tune_epochs": 1}
    tuned = train._run_tuning(cfg, None, None, None)

    assert tuned["model"]["learning_rate"] == 0.01
    assert tuned["model"]["optimizer"] == "sgd"
    assert cfg["model"]["learning_rate"] == 0.001  # original untouched (deep copy)


def test_tuning_stage_on_invokes_search_then_trains(tmp_path, monkeypatch) -> None:
    # stages.tuning: true must call _run_tuning, then still train end-to-end.
    train = _load_train_module()
    monkeypatch.setattr(train, "Fer2013Fetcher", _FakeFetcher)

    called = {}

    def _spy(cfg, train_ds, val_ds, class_weight):
        called["ran"] = True
        return cfg  # unchanged → normal training proceeds

    monkeypatch.setattr(train, "_run_tuning", _spy)

    history = train.run(_cfg(tmp_path, tuning=True))
    assert called.get("ran") is True
    assert "val_loss" in history.history


def test_tuning_stage_off_skips_search(tmp_path, monkeypatch) -> None:
    # default (tuning off) must NOT call _run_tuning.
    train = _load_train_module()
    monkeypatch.setattr(train, "Fer2013Fetcher", _FakeFetcher)

    def _boom(*args, **kwargs):
        raise AssertionError("_run_tuning should not run when stages.tuning is false")

    monkeypatch.setattr(train, "_run_tuning", _boom)
    train.run(_cfg(tmp_path))  # tuning defaults to False → no exception
