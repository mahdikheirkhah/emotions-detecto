"""Unit tests for the evaluation module (sklearn/matplotlib only — no TensorFlow)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import f1_score

from src.emotion_detector.models.evaluation import evaluate


class _FakeModel:
    """Returns pre-baked softmax scores so predictions are deterministic."""

    def __init__(self, probs: np.ndarray) -> None:
        self._probs = probs

    def predict(self, X, **kwargs):
        return self._probs


def _onehot(indices, n=7):
    m = np.zeros((len(indices), n), dtype=float)
    for row, i in enumerate(indices):
        m[row, i] = 1.0
    return m


def _cfg(tmp_path: Path, metrics=None, num_classes=7) -> dict:
    return {
        "model": {"num_classes": num_classes},
        "evaluation": {
            "metrics": metrics or ["accuracy", "f1_macro", "confusion_matrix"]
        },
        "paths": {"model_save_path": str(tmp_path / "model" / "final.keras")},
    }


# ---------------------------------------------------------------------------
# accuracy (hand-computed)
# ---------------------------------------------------------------------------


def test_accuracy_perfect(tmp_path: Path) -> None:
    y_true = np.array([0, 1, 2, 3])
    model = _FakeModel(_onehot([0, 1, 2, 3]))
    res = evaluate(model, np.zeros((4, 1)), y_true, _cfg(tmp_path), plot=False)
    assert res["accuracy"] == 1.0


def test_accuracy_four_of_five(tmp_path: Path) -> None:
    y_true = np.array([0, 0, 1, 1, 2])
    model = _FakeModel(_onehot([0, 1, 1, 1, 2]))  # 4/5 correct
    res = evaluate(model, np.zeros((5, 1)), y_true, _cfg(tmp_path), plot=False)
    assert res["accuracy"] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# macro-F1 (cross-checked against sklearn)
# ---------------------------------------------------------------------------


def test_f1_macro_matches_sklearn(tmp_path: Path) -> None:
    y_true = np.array([0, 0, 1, 1, 2])
    y_pred = [0, 1, 1, 1, 2]
    model = _FakeModel(_onehot(y_pred))
    res = evaluate(model, np.zeros((5, 1)), y_true, _cfg(tmp_path), plot=False)
    expected = f1_score(
        y_true, y_pred, labels=list(range(7)), average="macro", zero_division=0
    )
    assert res["f1_macro"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# confusion matrix (hand-built) & report
# ---------------------------------------------------------------------------


def test_confusion_matrix_values(tmp_path: Path) -> None:
    y_true = np.array([0, 0, 1])
    model = _FakeModel(_onehot([0, 1, 1]))  # (0→0), (0→1), (1→1)
    res = evaluate(model, np.zeros((3, 1)), y_true, _cfg(tmp_path), plot=False)
    cm = np.array(res["confusion_matrix"])
    assert cm.shape == (7, 7)  # square over all classes
    assert cm[0, 0] == 1 and cm[0, 1] == 1 and cm[1, 1] == 1


def test_classification_report_has_emotion_names(tmp_path: Path) -> None:
    y_true = np.array([0, 1, 3])
    model = _FakeModel(_onehot([0, 1, 3]))
    res = evaluate(model, np.zeros((3, 1)), y_true, _cfg(tmp_path), plot=False)
    report = res["classification_report"]
    assert "Angry" in report and "Happy" in report
    assert "precision" in report["Angry"]


# ---------------------------------------------------------------------------
# label formats, metric selection, plotting
# ---------------------------------------------------------------------------


def test_accepts_one_hot_labels(tmp_path: Path) -> None:
    preds = _onehot([0, 1, 2])
    res = evaluate(
        _FakeModel(preds), np.zeros((3, 1)), preds.copy(), _cfg(tmp_path), plot=False
    )
    assert res["accuracy"] == 1.0  # one-hot y_test argmax-decoded correctly


def test_only_requested_metrics_computed(tmp_path: Path) -> None:
    model = _FakeModel(_onehot([0, 1]))
    res = evaluate(
        model,
        np.zeros((2, 1)),
        np.array([0, 1]),
        _cfg(tmp_path, metrics=["accuracy"]),
        plot=False,
    )
    assert "accuracy" in res
    assert "f1_macro" not in res
    assert "confusion_matrix" not in res


def test_confusion_matrix_png_saved(tmp_path: Path) -> None:
    model = _FakeModel(_onehot([0, 1, 2]))
    res = evaluate(model, np.zeros((3, 1)), np.array([0, 1, 2]), _cfg(tmp_path))
    assert Path(res["confusion_matrix_path"]).exists()


def test_plot_false_skips_png(tmp_path: Path) -> None:
    model = _FakeModel(_onehot([0, 1, 2]))
    res = evaluate(
        model, np.zeros((3, 1)), np.array([0, 1, 2]), _cfg(tmp_path), plot=False
    )
    assert "confusion_matrix_path" not in res
