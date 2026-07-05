"""Unit tests for config-driven training callbacks (skipped without TensorFlow)."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("tensorflow")

from tensorflow.keras import callbacks as cb

from src.emotion_detector.models.callbacks import build_callbacks


def _cfg(tmp_path: Path, early_stopping=True, monitor="val_loss") -> dict:
    return {
        "callbacks": {
            "monitor": monitor,
            "early_stopping": early_stopping,
            "early_stopping_patience": 10,
            "reduce_lr_patience": 5,
            "reduce_lr_factor": 0.5,
            "min_lr": 1e-6,
        },
        "paths": {"model_save_path": str(tmp_path / "model" / "final.keras")},
    }


def _by_type(callbacks, cls):
    return next((c for c in callbacks if isinstance(c, cls)), None)


# ---------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------

def test_returns_three_callbacks(tmp_path: Path) -> None:
    callbacks = build_callbacks(_cfg(tmp_path))
    types = [type(c).__name__ for c in callbacks]
    assert types == ["EarlyStopping", "ModelCheckpoint", "ReduceLROnPlateau"]


def test_creates_model_directory(tmp_path: Path) -> None:
    build_callbacks(_cfg(tmp_path))
    assert (tmp_path / "model").is_dir()


# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------

def test_early_stopping_config(tmp_path: Path) -> None:
    es = _by_type(build_callbacks(_cfg(tmp_path)), cb.EarlyStopping)
    assert es.patience == 10
    assert es.restore_best_weights is True
    assert es.monitor == "val_loss"


def test_early_stopping_can_be_ablated(tmp_path: Path) -> None:
    callbacks = build_callbacks(_cfg(tmp_path, early_stopping=False))
    assert _by_type(callbacks, cb.EarlyStopping) is None
    assert len(callbacks) == 2  # only checkpoint + reduce-lr remain


# ---------------------------------------------------------------------------
# ModelCheckpoint
# ---------------------------------------------------------------------------

def test_model_checkpoint_config(tmp_path: Path) -> None:
    mc = _by_type(build_callbacks(_cfg(tmp_path)), cb.ModelCheckpoint)
    assert mc.save_best_only is True
    assert mc.filepath.endswith("final.keras")
    assert mc.monitor == "val_loss"


# ---------------------------------------------------------------------------
# ReduceLROnPlateau
# ---------------------------------------------------------------------------

def test_reduce_lr_config(tmp_path: Path) -> None:
    rl = _by_type(build_callbacks(_cfg(tmp_path)), cb.ReduceLROnPlateau)
    assert rl.factor == 0.5
    assert rl.patience == 5
    assert rl.min_lr == 1e-6
    assert rl.monitor == "val_loss"


# ---------------------------------------------------------------------------
# monitor is config-driven; missing key raises
# ---------------------------------------------------------------------------

def test_monitor_is_config_driven(tmp_path: Path) -> None:
    callbacks = build_callbacks(_cfg(tmp_path, monitor="val_accuracy"))
    for c in callbacks:
        assert c.monitor == "val_accuracy"


def test_missing_config_key_raises(tmp_path: Path) -> None:
    bad = {
        "callbacks": {"monitor": "val_loss"},  # missing patience/factor/min_lr
        "paths": {"model_save_path": str(tmp_path / "m.keras")},
    }
    with pytest.raises(KeyError, match="Missing callbacks/paths config key"):
        build_callbacks(bad)
