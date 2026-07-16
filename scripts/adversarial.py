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
    AttackResult,
    keras_attack_functions,
    probabilities_report,
    run_attack,
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


def save_attack_artifacts(cfg: dict, result: AttackResult) -> Dict[str, Any]:
    """Save the original|perturbation|adversarial figure + the adversarial ``.npy``.

    No TF. The perturbation is amplified for *display only* (its real values are ~eps,
    near invisible) so the middle panel shows what changed. Fake result in tests.
    """
    adv = cfg["adversarial"]
    comparison_path = Path(adv["comparison_path"])
    array_path = Path(adv["adversarial_array_path"])
    for path in (comparison_path, array_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    src_label = adv["target_class"]
    tgt_label = adv["attack_target_class"]
    src_idx = target_index(src_label)
    tgt_idx = target_index(tgt_label)

    adv_img = np.asarray(result.adversarial).reshape(48, 48)
    pert = np.asarray(result.perturbation).reshape(48, 48)
    original_img = adv_img - pert  # adversarial - perturbation == the source image
    pert_view = 0.5 + pert / (
        2 * (np.abs(pert).max() + 1e-8)
    )  # centre at grey, amplify

    orig_conf = float(np.asarray(result.original_probs).reshape(-1)[src_idx])
    adv_conf = float(np.asarray(result.adversarial_probs).reshape(-1)[tgt_idx])

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.4))
    axes[0].imshow(original_img, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"original: {src_label} {orig_conf:.0%}")
    axes[1].imshow(pert_view, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("perturbation (amplified)")
    axes[2].imshow(adv_img, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"adversarial: {tgt_label} {adv_conf:.0%}")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(comparison_path, dpi=120)
    plt.close(fig)

    np.save(array_path, adv_img[..., np.newaxis].astype(np.float32))
    return {
        "comparison_path": str(comparison_path),
        "adversarial_array_path": str(array_path),
    }


def run_and_save_attack(cfg: dict, model: Any, x_source: NDArray) -> AttackResult:
    """Run the configured FGSM/BIM attack on *x_source* and save the comparison."""
    adv = cfg["adversarial"]
    attack_target = target_index(adv["attack_target_class"])
    grad_fn, predict_fn = keras_attack_functions(model)
    result = run_attack(
        x_source,
        attack_target,
        grad_fn,
        predict_fn,
        epsilon=adv["epsilon"],
        attack_type=adv.get("attack_type", "fgsm"),
        iterations=adv.get("iterations", 10),
        step_size=adv.get("step_size"),
    )
    saved = save_attack_artifacts(cfg, result)
    linf = float(np.abs(result.perturbation).max())
    logger.info(
        f"Attack {'flipped' if result.success else 'did NOT flip'} to "
        f"{adv['attack_target_class']} in {result.iterations} step(s); "
        f"L-inf={linf:.4f} (budget {adv['epsilon']}). → {saved['comparison_path']}"
    )
    return result


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

    # Part 1 (#57): find the most confident source of the source class.
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

    # Part 2 (#58): attack that source toward the flip target.
    result = run_and_save_attack(cfg, model, images_norm[summary["index"]])

    orig = probabilities_report(result.original_probs)
    adv_probs = probabilities_report(result.adversarial_probs)
    src, tgt = (
        cfg["adversarial"]["target_class"],
        cfg["adversarial"]["attack_target_class"],
    )
    print(f"Original:    {src} {orig[src]:.0%}")
    print(
        f"Adversarial: {tgt} {adv_probs[tgt]:.0%} "
        f"({'flipped' if result.success else 'NOT flipped — raise epsilon'})"
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
