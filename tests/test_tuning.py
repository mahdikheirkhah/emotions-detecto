"""Unit tests for the hyperparameter-tuning scaffolding (Issue #43).

The heart of #43 — *the search space is built from config arrays correctly* — is
tested with a plain fake ``hp`` object, so it runs with **no ``keras_tuner`` and no
TensorFlow**. The tuner-dispatch and model-build tests are guarded by
``importorskip`` (they need the real ``keras_tuner`` / TF).
"""

from __future__ import annotations

import pytest

from src.emotion_detector.models.tuning import (
    apply_hyperparameters,
    build_hypermodel,
    make_tuner,
    sample_hyperparameters,
    search_space_size,
)


class _RecordingHp:
    """Fake Keras-Tuner ``hp``: records each ``Choice`` and returns the first value."""

    def __init__(self) -> None:
        self.registered: dict = {}

    def Choice(self, name, values):  # noqa: N802 — mirrors keras_tuner's API
        self.registered[name] = list(values)
        return values[0]


def _cfg(tmp_path=None, strategy="random", monitor="val_loss") -> dict:
    return {
        "global": {"seed": 42},
        "preprocessing": {"image_size": 48, "grayscale": True},
        "model": {
            "architecture": "simple_cnn",
            "optimizer": "adam",
            "learning_rate": 0.001,
            "batch_size": 64,
            "epochs": 1,
            "dropout_rate": 0.5,
            "num_conv_blocks": 3,
            "filters_start": 32,
            "kernel_size": 3,
            "convs_per_block": 2,
            "loss": "categorical_crossentropy",
            "num_classes": 7,
        },
        "callbacks": {"monitor": monitor},
        "tuning": {
            "strategy": strategy,
            "max_trials": 10,
            "executions_per_trial": 1,
            "hyperband_max_epochs": 4,
            "tuning_dir": str(tmp_path) if tmp_path else "logs/tuning/",
            "project_name": "fer_tuning_test",
            "search_space": {
                "learning_rate": [0.01, 0.001, 0.0001],
                "batch_size": [32, 64, 128],
                "dropout_rate": [0.3, 0.4, 0.5],
                "filters_start": [32, 64],
                "optimizer": ["adam", "sgd", "rmsprop"],
            },
        },
    }


# ---------------------------------------------------------------------------
# search space is built from the config arrays (the core of #43)
# ---------------------------------------------------------------------------


def test_search_space_registers_config_arrays_verbatim() -> None:
    hp = _RecordingHp()
    cfg = _cfg()
    sample_hyperparameters(hp, cfg)
    space = cfg["tuning"]["search_space"]
    # every config array became a search dimension, values unchanged
    assert hp.registered == {k: list(v) for k, v in space.items()}


def test_sampled_values_are_drawn_from_the_arrays() -> None:
    cfg = _cfg()
    sampled = sample_hyperparameters(_RecordingHp(), cfg)
    space = cfg["tuning"]["search_space"]
    assert set(sampled) == set(space)
    for knob, value in sampled.items():
        assert value in space[knob]


def test_adding_a_knob_to_config_extends_the_space() -> None:
    cfg = _cfg()
    cfg["tuning"]["search_space"]["kernel_size"] = [3, 5]  # config-only change
    hp = _RecordingHp()
    sample_hyperparameters(hp, cfg)
    assert hp.registered["kernel_size"] == [3, 5]


def test_search_space_size_is_the_grid_product() -> None:
    # 3 * 3 * 3 * 2 * 3
    assert search_space_size(_cfg()) == 162


# ---------------------------------------------------------------------------
# applying a sampled config overrides the model.* defaults
# ---------------------------------------------------------------------------


def test_apply_overrides_model_defaults_without_mutating_original() -> None:
    cfg = _cfg()
    sampled = {
        "learning_rate": 0.01,
        "batch_size": 128,
        "dropout_rate": 0.3,
        "filters_start": 64,
        "optimizer": "sgd",
    }
    tuned = apply_hyperparameters(cfg, sampled)
    for knob, value in sampled.items():
        assert tuned["model"][knob] == value
    # deep copy → the original config is untouched
    assert cfg["model"]["learning_rate"] == 0.001
    assert cfg["model"]["optimizer"] == "adam"


def test_apply_unknown_knob_fails_loud() -> None:
    with pytest.raises(KeyError):
        apply_hyperparameters(_cfg(), {"not_a_model_key": 1})


# ---------------------------------------------------------------------------
# tuner dispatch + objective (need keras_tuner)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "strategy, cls_name",
    [
        ("grid", "GridSearch"),
        ("random", "RandomSearch"),
        ("bayesian", "BayesianOptimization"),
        ("hyperband", "Hyperband"),
    ],
)
def test_make_tuner_dispatches_by_strategy(tmp_path, strategy, cls_name) -> None:
    pytest.importorskip("keras_tuner")
    cfg = _cfg(tmp_path=tmp_path, strategy=strategy)
    tuner = make_tuner(cfg, hypermodel=lambda hp: None)  # dummy: not called at build
    assert type(tuner).__name__ == cls_name


def test_make_tuner_unknown_strategy_fails_loud(tmp_path) -> None:
    pytest.importorskip("keras_tuner")
    cfg = _cfg(tmp_path=tmp_path, strategy="teleport")
    with pytest.raises(ValueError):
        make_tuner(cfg, hypermodel=lambda hp: None)


def test_objective_mirrors_callbacks_monitor(tmp_path) -> None:
    pytest.importorskip("keras_tuner")
    from src.emotion_detector.models.tuning import _tuner_objective

    loss_obj = _tuner_objective(_cfg(monitor="val_loss"))
    assert loss_obj.name == "val_loss" and loss_obj.direction == "min"
    acc_obj = _tuner_objective(_cfg(monitor="val_accuracy"))
    assert acc_obj.name == "val_accuracy" and acc_obj.direction == "max"


# ---------------------------------------------------------------------------
# build_hypermodel actually applies the sampled values (needs TensorFlow)
# ---------------------------------------------------------------------------


def test_build_hypermodel_applies_sampled_overrides(tmp_path) -> None:
    pytest.importorskip("tensorflow")

    class _FixedHp:
        """Always picks a chosen value so the override is deterministic."""

        def Choice(self, name, values):  # noqa: N802
            return {"optimizer": "sgd", "learning_rate": 0.01}.get(name, values[0])

    model = build_hypermodel(_FixedHp(), _cfg(tmp_path=tmp_path))
    assert model.optimizer.__class__.__name__ == "SGD"  # optimizer override took
    assert float(model.optimizer.learning_rate) == pytest.approx(0.01)  # lr override
    assert model.count_params() > 0  # compiled, real model
