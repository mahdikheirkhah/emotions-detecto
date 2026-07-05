"""Config-driven model compilation — optimizer, loss, and metrics via dispatch.

Small file, big consequences: this wires in the **gradient-update rule**
(optimizer + learning rate) and the **training objective** (loss), plus the
metrics we *report*. Everything is a ``config.yaml`` value so optimizer / LR /
loss are prime tuning knobs (CONTRIBUTING §3).

TensorFlow is imported lazily so importing this module is cheap.
"""
from __future__ import annotations

from typing import Any, List

from src.emotion_detector.utils.dispatch import dispatch
from src.emotion_detector.utils.logging import logger

# SGD momentum default (config has no separate knob; 0.9 is the standard value).
_SGD_MOMENTUM = 0.9


def build_optimizer(cfg: dict) -> Any:
    """Build the optimizer selected by ``model.optimizer`` at ``model.learning_rate``.

    The learning rate is the single most important hyperparameter, so it is
    config-driven; each optimizer adapts the step differently (SGD+momentum uses
    a fixed global rate with velocity; Adam/RMSprop adapt per-parameter).

    Args:
        cfg: Loaded config dict (reads ``model.optimizer``, ``model.learning_rate``).

    Returns:
        A ``keras.optimizers.Optimizer`` instance.

    Raises:
        ValueError: if ``model.optimizer`` is not a known option.
        KeyError:   if a required config key is missing.
    """
    from tensorflow import keras

    m = cfg["model"]
    lr = m["learning_rate"]
    registry = {
        "adam": lambda: keras.optimizers.Adam(learning_rate=lr),
        "sgd": lambda: keras.optimizers.SGD(learning_rate=lr, momentum=_SGD_MOMENTUM),
        "rmsprop": lambda: keras.optimizers.RMSprop(learning_rate=lr),
    }
    return dispatch(m["optimizer"], registry)


def build_loss(cfg: dict) -> Any:
    """Return the loss selected by ``model.loss``.

    ``categorical_crossentropy`` is the natural loss for one-hot multi-class
    targets with a softmax output. ``focal_loss`` is a valid config option but
    lands in a later issue.

    Raises:
        ValueError: if ``model.loss`` is not supported yet.
    """
    name = cfg["model"]["loss"]
    if name == "categorical_crossentropy":
        return "categorical_crossentropy"
    raise ValueError(
        f"Unsupported model.loss '{name}'. Only 'categorical_crossentropy' is "
        "implemented so far (focal_loss is a later issue)."
    )


def build_metrics(cfg: dict) -> List[str]:
    """Metrics to *report* during training (distinct from the optimized loss).

    Accuracy is tracked here; macro-F1 / per-class recall are computed in the
    evaluation stage (data.md §3.2), not as Keras training metrics.
    """
    return ["accuracy"]


def compile_model(model: Any, cfg: dict) -> Any:
    """Compile *model* in place with the config-selected optimizer/loss/metrics.

    Args:
        model: An assembled (uncompiled) ``keras.Model``.
        cfg:   Loaded config dict.

    Returns:
        The same *model*, now compiled and ready for ``.fit()``.
    """
    optimizer = build_optimizer(cfg)
    loss = build_loss(cfg)
    metrics = build_metrics(cfg)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    logger.info(
        f"Compiled — optimizer={cfg['model']['optimizer']} "
        f"(lr={cfg['model']['learning_rate']}), loss={cfg['model']['loss']}, "
        f"metrics={metrics}"
    )
    return model
