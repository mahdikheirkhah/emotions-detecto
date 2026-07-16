"""Adversarial bonus, part 1 (#57): pick a Happy-with->90% source image.

Scans a dataset split with the trained model and picks the single image it most
confidently classifies as ``adversarial.target_class`` (default *Happy*) above the
confidence threshold -- the starting point for the FGSM attack in #58. Saves three:

  * ``source_image.png``          -- the chosen 48x48 face (viewable), titled
  * ``source_array.npy``          -- its normalized ``(48, 48, 1)`` input (#58 perturbs)
  * ``source_probabilities.json`` -- the full 7-class softmax, for the record

The **FGSM concept** this sets up (gradient w.r.t. the *input*, ``x + eps*sign(grad)``,
and why imperceptible changes fool the net) is written up in
``src/emotion_detector/adversarial.py``. Target class, threshold, split, and ``epsilon``
are all ``config.yaml`` values (Ablation section 3); the run is seeded + reproducible.

    python scripts/adversarial.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib

matplotlib.use("Agg")  # headless: write the figure to a file, never open a window
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.emotion_detector.adversarial import (
    probabilities_report,
    select_target_image,
    target_index,
)
from src.emotion_detector.data.cleaning import clean_dataset
from src.emotion_detector.data.fer2013 import Fer2013Fetcher
from src.emotion_detector.data.preprocessing import build_normalizer
from src.emotion_detector.data.splits import make_splits
from src.emotion_detector.models.labels import FER_EMOTIONS
from src.emotion_detector.utils.config import load_config
from src.emotion_detector.utils.logging import logger, setup_logging
from src.emotion_detector.utils.seeding import set_global_seed

_SPLITS = ("Training", "PublicTest", "PrivateTest")


def find_and_save(
    cfg: dict,
    probabilities: NDArray,
    images_uint8: NDArray,
    images_norm: NDArray,
) -> Dict[str, Any]:
    """Select the source image from precomputed probs and write the three artifacts.

    No TensorFlow here: the model's ``probabilities`` are passed in, so this is tested
    with fakes. *images_uint8* are the original ``(N, 48, 48)`` frames (for the PNG);
    *images_norm* are the ``(N, 48, 48, 1)`` model inputs (saved for #58).

    Returns a summary dict (chosen index, confidence, label, output paths).
    """
    adv = cfg["adversarial"]
    class_index = target_index(adv["target_class"])
    idx, confidence = select_target_image(
        probabilities, class_index, adv["source_confidence_threshold"]
    )
    report = probabilities_report(probabilities[idx])

    image_path = Path(adv["source_image_path"])
    array_path = Path(adv["source_array_path"])
    probs_path = Path(adv["source_probs_path"])
    for path in (image_path, array_path, probs_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(np.asarray(images_uint8[idx]).reshape(48, 48), cmap="gray")
    ax.set_title(f"{adv['target_class']}: {confidence:.1%}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(image_path, dpi=120)
    plt.close(fig)

    np.save(array_path, np.asarray(images_norm[idx], dtype=np.float32))
    probs_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return {
        "index": idx,
        "confidence": confidence,
        "label": adv["target_class"],
        "image_path": str(image_path),
        "array_path": str(array_path),
        "probs_path": str(probs_path),
    }


def _load_split(cfg: dict) -> Tuple[NDArray, NDArray]:
    """Return ``(images_uint8, images_norm)`` for the configured ``scan_split``.

    Fits the normalizer exactly as training did — on the cleaned **train** split — then
    applies it to the split being scanned (never re-fit on it). No TF needed.
    """
    set_global_seed(cfg["global"]["seed"])  # deterministic split → reproducible source
    fetcher = Fer2013Fetcher(cfg)
    images, labels, usage = [], [], []
    for split in _SPLITS:
        Xi, yi = fetcher.fetch(split)
        images.append(Xi)
        labels.append(yi)
        usage.append(np.full(len(yi), split))
    X = np.concatenate(images)
    y = np.concatenate(labels)
    usage = np.concatenate(usage)

    X_train, y_train, _, _, X_test, _ = make_splits(cfg, X, y, usage)
    X_train, _ = clean_dataset(cfg, X_train, y_train)
    normalizer = build_normalizer(cfg).fit(X_train)

    scan = cfg["adversarial"].get("scan_split", "test")
    images_uint8 = X_train if scan == "train" else X_test
    images_norm = normalizer.transform(images_uint8)
    return images_uint8, images_norm[..., np.newaxis].astype(np.float32)


def _load_keras_model(path: str) -> Any:
    """Load a saved ``.keras`` model (import kept local + patchable for tests)."""
    from tensorflow.keras.models import load_model

    return load_model(path)


def main(config_path: str = "config.yaml") -> None:
    cfg = load_config(config_path)
    setup_logging(cfg)

    model_path = cfg["paths"]["model_save_path"]
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"No trained model at {model_path}. Run scripts/train.py first."
        )
    model = _load_keras_model(model_path)

    images_uint8, images_norm = _load_split(cfg)
    split_name = cfg["adversarial"].get("scan_split", "test")
    logger.info(
        f"Scanning {len(images_norm):,} '{split_name}' images for a confident "
        f"{cfg['adversarial']['target_class']} source ..."
    )
    probabilities = model.predict(images_norm, verbose=0)

    summary = find_and_save(cfg, probabilities, images_uint8, images_norm)
    logger.info(
        f"Chosen source: index {summary['index']} — "
        f"{summary['label']} {summary['confidence']:.1%} → {summary['image_path']}"
    )
    print(
        f"Source image: {summary['label']} at {summary['confidence']:.0%} "
        f"(saved to {summary['image_path']})"
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
