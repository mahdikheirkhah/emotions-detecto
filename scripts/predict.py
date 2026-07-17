"""Evaluate the trained model on the held-out test set (audit entrypoint).

Loads the trained ``.keras``, reproduces the **exact training preprocessing** on the
test split (fitted on train, never re-fit on test), and prints the line the audit
greps::

    Accuracy on test set: XX%

The test split is the ``Usage != Training`` rows (PublicTest + PrivateTest =
test_with_emotions.csv) from the same seeded split the trainer used, so this is a
single, final, untouched measurement — no tuning against it.

Which model it loads is transfer-aware (``resolve_model_path``): a ``transfer_*``
architecture evaluates the separate ``pretrained_*`` model. Pass an alternate config to
score the transfer model built by ``config_transfer.yaml``::

    python ./scripts/predict.py                    # the from-scratch model
    python ./scripts/predict.py config_transfer.yaml   # the pretrained (transfer) model
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.emotion_detector.data.cleaning import clean_dataset
from src.emotion_detector.data.fer2013 import Fer2013Fetcher
from src.emotion_detector.data.pipeline import to_tensors
from src.emotion_detector.data.preprocessing import build_normalizer
from src.emotion_detector.data.splits import make_splits
from src.emotion_detector.models.classifier import resolve_model_path
from src.emotion_detector.models.evaluation import evaluate
from src.emotion_detector.utils.config import load_config
from src.emotion_detector.utils.logging import logger, setup_logging
from src.emotion_detector.utils.seeding import set_global_seed

_SPLITS = ("Training", "PublicTest", "PrivateTest")


def _load_all_rows(cfg: dict):
    fetcher = Fer2013Fetcher(cfg)
    images, labels, usage = [], [], []
    for split in _SPLITS:
        Xi, yi = fetcher.fetch(split)
        images.append(Xi)
        labels.append(yi)
        usage.append(np.full(len(yi), split))
    return np.concatenate(images), np.concatenate(labels), np.concatenate(usage)


def score_test_set(cfg: dict, model: Any) -> dict:
    """Reproduce training preprocessing on the test split and evaluate *model*.

    The normalizer is fit on the **cleaned training split** (the same data, seeded
    → identical stats) and only *applied* to test — never re-fit on test (that
    would leak the evaluation distribution into the transform).

    Returns the metrics dict from ``models.evaluation.evaluate``.
    """
    set_global_seed(cfg["global"]["seed"])  # deterministic split → same test rows

    X, y, usage = _load_all_rows(cfg)
    X_train, y_train, _, _, X_test, y_test = make_splits(cfg, X, y, usage)

    # Fit the normalizer exactly as training did: on the cleaned TRAIN split.
    X_train, _ = clean_dataset(cfg, X_train, y_train)
    normalizer = build_normalizer(cfg).fit(X_train)
    X_test = normalizer.transform(X_test)

    # Add the channel axis the model expects (labels unused here).
    X_test, _ = to_tensors(X_test, y_test, num_classes=cfg["model"]["num_classes"])

    logger.info(f"Scoring {len(y_test):,} held-out test images.")
    return evaluate(model, X_test, y_test, cfg, plot=True)


def _load_keras_model(path: str) -> Any:
    """Load a saved ``.keras`` model (import kept local + patchable for tests)."""
    from tensorflow.keras.models import load_model

    return load_model(path)


def main(config_path: str = "config.yaml") -> None:
    cfg = load_config(config_path)
    setup_logging(cfg)

    model_path = resolve_model_path(cfg)  # pretrained_* for a transfer_* architecture
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"No trained model at {model_path}. Run scripts/train.py first."
        )
    model = _load_keras_model(model_path)

    results = score_test_set(cfg, model)
    acc = results["accuracy"]

    if "f1_macro" in results:
        logger.info(f"macro-F1: {results['f1_macro']:.4f}")
    if "confusion_matrix_path" in results:
        logger.info(f"confusion matrix → {results['confusion_matrix_path']}")

    # print() is intentional in this script — the audit greps this exact line.
    print(f"Accuracy on test set: {acc:.0%}")


if __name__ == "__main__":
    # Optional config path, e.g. `python scripts/predict.py config_transfer.yaml` to
    # evaluate the pretrained (transfer) model instead of the from-scratch one.
    main(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
