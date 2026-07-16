"""Unit tests for config-driven training callbacks (skipped without TensorFlow)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("tensorflow")

from tensorflow.keras import callbacks as cb

from src.emotion_detector.models.callbacks import build_callbacks


def _cfg(tmp_path, early_stopping=True, monitor="val_loss", tensorboard=True) -> dict:
    return {
        "callbacks": {
            "monitor": monitor,
            "early_stopping": early_stopping,
            "tensorboard": tensorboard,
            "early_stopping_patience": 10,
            "reduce_lr_patience": 5,
            "reduce_lr_factor": 0.5,
            "min_lr": 1e-6,
        },
        "model": {"architecture": "vgg_small"},
        "preprocessing": {"normalization": "rescale"},
        "stages": {"augmentation": True, "cleaning": True},
        "paths": {
            "model_save_path": str(tmp_path / "model" / "final.keras"),
            "tensorboard_dir": str(tmp_path / "tb"),
        },
    }


def _by_type(callbacks, cls):
    return next((c for c in callbacks if isinstance(c, cls)), None)


# ---------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------


def test_returns_all_callbacks(tmp_path: Path) -> None:
    callbacks = build_callbacks(_cfg(tmp_path))
    types = [type(c).__name__ for c in callbacks]
    assert types == [
        "EarlyStopping",
        "ModelCheckpoint",
        "ReduceLROnPlateau",
        "TensorBoard",
    ]


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
    # checkpoint + reduce-lr + tensorboard remain
    assert len(callbacks) == 3


# ---------------------------------------------------------------------------
# TensorBoard
# ---------------------------------------------------------------------------


def test_tensorboard_present_and_logs_under_root(tmp_path: Path) -> None:
    tb = _by_type(build_callbacks(_cfg(tmp_path)), cb.TensorBoard)
    assert tb is not None
    assert str(tmp_path / "tb") in tb.log_dir  # per-run dir under the config root


def test_tensorboard_run_dir_created(tmp_path: Path) -> None:
    build_callbacks(_cfg(tmp_path))
    # the timestamped run dir exists under the tensorboard root
    assert any((tmp_path / "tb").iterdir())


def test_tensorboard_can_be_ablated(tmp_path: Path) -> None:
    callbacks = build_callbacks(_cfg(tmp_path, tensorboard=False))
    assert _by_type(callbacks, cb.TensorBoard) is None


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
        if hasattr(c, "monitor"):  # TensorBoard has no monitor
            assert c.monitor == "val_accuracy"


def test_missing_config_key_raises(tmp_path: Path) -> None:
    bad = {
        "callbacks": {"monitor": "val_loss"},  # missing patience/factor/min_lr
        "paths": {"model_save_path": str(tmp_path / "m.keras")},
    }
    with pytest.raises(KeyError, match="Missing callbacks/paths config key"):
        build_callbacks(bad)
