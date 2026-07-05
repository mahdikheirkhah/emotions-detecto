"""Unit tests for reading the trials table + selecting the winner (Issue #44).

Uses a fake tuner mimicking the ``keras_tuner`` oracle API, so it runs with pandas
only — no ``keras_tuner`` and no TensorFlow.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.emotion_detector.models.tuning import (
    best_hyperparameters,
    results_table,
    save_results_table,
)


class _FakeHP:
    def __init__(self, values: dict) -> None:
        self.values = values


class _FakeTrial:
    def __init__(self, trial_id: str, score: float, values: dict) -> None:
        self.trial_id = trial_id
        self.score = score
        self.hyperparameters = _FakeHP(values)


class _FakeOracle:
    """Mimics keras_tuner's oracle: get_best_trials returns best (highest) first."""

    def __init__(self, trials) -> None:
        self.trials = {t.trial_id: t for t in trials}
        self._sorted = sorted(trials, key=lambda t: t.score, reverse=True)

    def get_best_trials(self, num_trials):
        return self._sorted[:num_trials]


class _FakeTuner:
    def __init__(self, trials) -> None:
        self.oracle = _FakeOracle(trials)

    def get_best_hyperparameters(self, num_trials=1):
        return [t.hyperparameters for t in self.oracle.get_best_trials(num_trials)]


def _tuner():
    return _FakeTuner(
        [
            _FakeTrial("a", 0.55, {"learning_rate": 0.01, "optimizer": "sgd"}),
            _FakeTrial("b", 0.62, {"learning_rate": 0.001, "optimizer": "adam"}),
            _FakeTrial("c", 0.48, {"learning_rate": 0.0001, "optimizer": "rmsprop"}),
        ]
    )


# ---------------------------------------------------------------------------
# results_table
# ---------------------------------------------------------------------------


def test_results_table_one_row_per_trial_sorted_best_first() -> None:
    df = results_table(_tuner())
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df["rank"]) == [1, 2, 3]
    # best score (0.62, trial "b") ranks first
    assert df.iloc[0]["trial_id"] == "b"
    assert df.iloc[0]["score"] == 0.62
    assert list(df["score"]) == [0.62, 0.55, 0.48]  # descending


def test_results_table_has_a_column_per_hyperparameter() -> None:
    df = results_table(_tuner())
    for col in ("rank", "trial_id", "score", "learning_rate", "optimizer"):
        assert col in df.columns
    assert df.iloc[0]["optimizer"] == "adam"  # winner's value


# ---------------------------------------------------------------------------
# best_hyperparameters
# ---------------------------------------------------------------------------


def test_best_hyperparameters_returns_top_trial_values() -> None:
    best = best_hyperparameters(_tuner())
    assert best == {"learning_rate": 0.001, "optimizer": "adam"}


# ---------------------------------------------------------------------------
# save_results_table
# ---------------------------------------------------------------------------


def test_save_results_table_writes_csv_and_json(tmp_path: Path) -> None:
    df = results_table(_tuner())
    cfg = {"paths": {"results_dir": str(tmp_path)}}
    paths = save_results_table(df, cfg)

    assert Path(paths["csv"]).exists()
    assert Path(paths["json"]).exists()
    # round-trips: the CSV reloads to the same ranked order
    reloaded = pd.read_csv(paths["csv"])
    assert list(reloaded["trial_id"]) == ["b", "a", "c"]
