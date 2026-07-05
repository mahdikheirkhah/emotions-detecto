"""Unit tests for the learning-curves plot (matplotlib only — no TensorFlow)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parent.parent / "scripts" / "validation_loss_accuracy.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("vla_entrypoint", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_vla = _load()


def _overfit_history() -> dict:
    """Val loss bottoms at epoch 6 (index 5) then rises — classic overfitting."""
    return {
        "loss": [1.8, 1.3, 1.0, 0.8, 0.62, 0.5, 0.42, 0.35, 0.30, 0.26],
        "val_loss": [1.6, 1.2, 0.95, 0.82, 0.74, 0.70, 0.72, 0.77, 0.85, 0.95],
        "accuracy": [0.35, 0.52, 0.61, 0.68, 0.73, 0.78, 0.81, 0.84, 0.86, 0.89],
        "val_accuracy": [0.40, 0.55, 0.63, 0.68, 0.71, 0.72, 0.71, 0.71, 0.70, 0.69],
    }


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

def test_plot_produces_png(tmp_path: Path) -> None:
    out = _vla.plot_learning_curves(_overfit_history(), tmp_path / "curves.png")
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_creates_parent_dir(tmp_path: Path) -> None:
    out = _vla.plot_learning_curves(_overfit_history(), tmp_path / "nested" / "c.png")
    assert out.exists()


def test_plot_loss_only_history(tmp_path: Path) -> None:
    # missing accuracy/val series must not crash
    out = _vla.plot_learning_curves(
        {"loss": [1.0, 0.5, 0.3]}, tmp_path / "loss_only.png"
    )
    assert out.exists()


def test_plot_missing_loss_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no 'loss' series"):
        _vla.plot_learning_curves({"accuracy": [0.5]}, tmp_path / "x.png")


# ---------------------------------------------------------------------------
# early-stopping epoch detection
# ---------------------------------------------------------------------------

def test_best_epoch_is_val_loss_minimum() -> None:
    assert _vla._best_epoch(_overfit_history()) == 5  # 0-based → epoch 6


def test_best_epoch_none_without_val_loss() -> None:
    assert _vla._best_epoch({"loss": [1.0, 0.5]}) is None
