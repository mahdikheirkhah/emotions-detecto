"""Config-driven hyperparameter search space + Keras-Tuner setup (scaffolding).

**Parameters vs hyperparameters:** the network's *weights* are **learned** by
gradient descent; *hyperparameters* (learning rate, batch size, dropout, #filters,
optimizer) are **chosen** — and this module searches for good choices.

The search **space** lives entirely in ``config.yaml`` under ``tuning.search_space``
as arrays of candidate values — the purest expression of the Ablation-Driven
Architecture (CONTRIBUTING §3): one array per knob, no space hardcoded in Python.
``sample_hyperparameters`` maps each array to a Keras-Tuner ``hp.Choice`` dimension;
``build_hypermodel`` overrides the matching ``model.*`` default with the sampled
value and builds a compiled model; ``make_tuner`` dispatches on ``tuning.strategy``.

**Searched on validation only, never test** (CONTRIBUTING §8): the tuner's objective
mirrors ``callbacks.monitor`` (``val_loss`` / ``val_accuracy``) so the final test
number stays an untouched, honest measurement. This issue (#43) is the scaffolding;
the actual ``tuner.search(...)`` run is #44.

``keras_tuner`` is imported lazily (only ``make_tuner`` needs it) so the search-space
logic stays testable with a plain fake ``hp``.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict

from src.emotion_detector.utils.dispatch import dispatch
from src.emotion_detector.utils.logging import logger


def sample_hyperparameters(hp: Any, cfg: dict) -> Dict[str, Any]:
    """Register one search dimension per ``tuning.search_space`` array and sample it.

    For each knob ``k`` with candidates ``[v0, v1, ...]`` this calls
    ``hp.Choice(k, [v0, v1, ...])`` (discrete arrays → ``hp.Choice``; the purest
    ablation form — a continuous knob could use ``hp.Float`` instead). The returned
    dict is ``{knob: sampled_value}`` for exactly the knobs listed in config, so
    adding/removing a search dimension is a config-only change.

    Args:
        hp:  A Keras-Tuner ``HyperParameters`` (or any object exposing
             ``Choice(name, values)`` — a fake in tests).
        cfg: Loaded config dict (reads ``tuning.search_space``).

    Returns:
        ``{knob: value}`` — the value the tuner picked for this trial.
    """
    space = cfg["tuning"]["search_space"]
    return {
        knob: hp.Choice(knob, list(candidates)) for knob, candidates in space.items()
    }


def apply_hyperparameters(cfg: dict, sampled: Dict[str, Any]) -> dict:
    """Return a deep copy of *cfg* with each sampled value written over ``model.<knob>``.

    Every search knob must name a real ``model.*`` key it overrides — a typo in
    ``tuning.search_space`` fails loud here rather than silently doing nothing.

    Raises:
        KeyError: if a searched knob has no matching ``model.<knob>`` default.
    """
    tuned = copy.deepcopy(cfg)
    model = tuned["model"]
    for knob, value in sampled.items():
        if knob not in model:
            raise KeyError(
                f"tuning.search_space knob '{knob}' has no matching model.{knob} "
                f"default to override. Known model keys: {sorted(model)}"
            )
        model[knob] = value
    return tuned


def build_hypermodel(hp: Any, cfg: dict) -> Any:
    """Keras-Tuner hypermodel fn: sample the space, override config, build a model.

    Passed to the tuner as ``lambda hp: build_hypermodel(hp, cfg)``. Each trial the
    tuner supplies a fresh ``hp``; we sample the knobs, splice them into a copied
    config, and reuse the existing ``build_model`` dispatch so the tuned model is
    identical to a normal build except for the searched values.

    Returns:
        A **compiled** ``keras.Model`` for this trial's hyperparameters.
    """
    sampled = sample_hyperparameters(hp, cfg)
    tuned_cfg = apply_hyperparameters(cfg, sampled)
    # Imported here (not at module top) to keep TensorFlow out of the import path
    # of the search-space logic above.
    from src.emotion_detector.models.builders import build_model

    return build_model(tuned_cfg, summary=False)


def _tuner_objective(cfg: dict) -> Any:
    """Build the Keras-Tuner objective from ``callbacks.monitor`` — validation only.

    ``val_loss`` is minimized, ``val_accuracy`` maximized; either way it is a
    *validation* metric, so tuning never sees the test set (CONTRIBUTING §8).
    """
    import keras_tuner as kt

    monitor = cfg["callbacks"]["monitor"]
    direction = "min" if monitor.endswith("loss") else "max"
    return kt.Objective(monitor, direction=direction)


def make_tuner(cfg: dict, hypermodel: Callable[[Any], Any] | None = None) -> Any:
    """Build the Keras-Tuner tuner selected by ``tuning.strategy`` (dispatch).

    Strategies trade coverage against cost: ``grid`` tries every combination
    (exhaustive), ``random`` draws ``max_trials`` points (best coverage per unit
    cost), ``bayesian`` models the objective to focus on promising regions, and
    ``hyperband`` runs successive halving — cheaply training many configs briefly
    and promoting only the strongest (``hyperband_max_epochs`` / ``factor``), so it
    ignores ``max_trials``.

    Args:
        cfg:        Loaded config dict (reads the ``tuning`` block + ``global.seed``).
        hypermodel: Optional override; defaults to ``build_hypermodel`` bound to *cfg*.

    Returns:
        An unfitted ``keras_tuner`` tuner ready for ``.search(...)`` in #44.

    Raises:
        ValueError: if ``tuning.strategy`` is not a known option.
    """
    import keras_tuner as kt

    t = cfg["tuning"]
    strategy = t["strategy"]
    if hypermodel is None:
        hypermodel = lambda hp: build_hypermodel(hp, cfg)  # noqa: E731

    # Args shared by every tuner; each strategy adds its own budget knob below.
    common: Dict[str, Any] = dict(
        hypermodel=hypermodel,
        objective=_tuner_objective(cfg),
        seed=cfg["global"]["seed"],
        executions_per_trial=t.get("executions_per_trial", 1),
        directory=t["tuning_dir"],
        project_name=t.get("project_name", "fer_tuning"),
        overwrite=True,
    )
    max_trials = t["max_trials"]
    registry = {
        "grid": lambda: kt.GridSearch(max_trials=max_trials, **common),
        "random": lambda: kt.RandomSearch(max_trials=max_trials, **common),
        "bayesian": lambda: kt.BayesianOptimization(max_trials=max_trials, **common),
        "hyperband": lambda: kt.Hyperband(
            max_epochs=t.get("hyperband_max_epochs", 30), factor=3, **common
        ),
    }
    tuner = dispatch(strategy, registry)

    space = t["search_space"]
    combos = 1
    for candidates in space.values():
        combos *= len(candidates)
    logger.info(
        f"Tuner '{strategy}' ready — {len(space)} knobs, {combos} grid combinations, "
        f"objective={common['objective'].name} ({common['objective'].direction}), "
        f"budget={'all combos' if strategy == 'grid' else f'{max_trials} trials'}."
    )
    return tuner


def search_space_size(cfg: dict) -> int:
    """Total number of hyperparameter combinations (the full grid size).

    Useful for sanity-checking cost before a run: ``random``/``bayesian`` sample a
    fraction of this, ``grid`` covers all of it.
    """
    combos = 1
    for candidates in cfg["tuning"]["search_space"].values():
        combos *= len(candidates)
    return combos
