"""Adversarial-attack helpers + the FGSM concept (bonus, #57 setup / #58 attack).

TF-free selection logic for the "hack the CNN" bonus, kept here so it is unit-tested
without a model. #57 uses it to pick a source image the model is very sure is *Happy*;
#58 will perturb that image to flip the prediction.

=================================================================================
FGSM -- the Fast Gradient Sign Method, in plain words
=================================================================================

An **adversarial example** is an input changed by a tiny, usually invisible amount
that nonetheless flips the prediction. The same 48x48 face that reads "Happy 99%" can,
after nudging each pixel by a hair, read "Sad 99%" while looking identical to a human.

**The key inversion: gradient w.r.t. the *input*, not the weights.** Training computes
d(loss)/d(weights) and steps the *weights* to cut loss on fixed data. FGSM freezes the
trained weights and instead computes d(loss)/d(pixels): with the model held constant, it
asks "how should each pixel change to *raise* the loss for the true class?" -- i.e. to
make the model more wrong. (Targeted variant: *descend* a chosen target class's loss
instead, steering the prediction to a specific emotion.)

**The step.** FGSM takes the **sign** of that input-gradient and moves every pixel a
fixed amount ``epsilon`` in that direction::

    x_adv = x + epsilon * sign( d(loss)/d(x) )

Using ``sign`` (not the raw gradient) caps the change per pixel at ``epsilon`` -- the
perturbation's L-infinity norm is exactly ``epsilon`` -- so a small ``epsilon`` keeps it
imperceptible while still moving *every* pixel.

**Why such tiny changes fool the net.** In the input's very high dimensions the model
behaves *locally almost linearly*. A logit is roughly ``w . x``, so perturbing by
``d = epsilon * sign(w)`` shifts it by about ``epsilon * sum(|w_i|)``. Each pixel adds
only ``epsilon * |w_i|`` (negligible alone), but summed over thousands of pixels *all
pushed the same helpful way* the total logit swing is large. The attack wins not by one
big change but by many minuscule, perfectly-coordinated ones.

``target_class``, the source-confidence threshold, and ``epsilon`` are ``config.yaml``
values (Ablation section 3); nothing here is hardcoded.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from src.emotion_detector.models.labels import FER_EMOTIONS


def target_index(target_class: str, labels: Sequence[str] = FER_EMOTIONS) -> int:
    """The class index for ``target_class`` (e.g. ``"Happy"`` -> 3).

    Raises:
        ValueError: if *target_class* is not a known emotion label.
    """
    if target_class not in labels:
        raise ValueError(f"target_class '{target_class}' is not one of {list(labels)}.")
    return list(labels).index(target_class)


def select_target_image(
    probabilities: NDArray, class_index: int, threshold: float
) -> Tuple[int, float]:
    """Pick the row the model most confidently calls *class_index*, above *threshold*.

    Scans a batch of softmax vectors and returns the single best attack source: the
    image whose predicted (argmax) class is *class_index* with the **highest** score
    over *threshold*. Deterministic (highest first, lowest row index on ties).

    Args:
        probabilities: ``(N, C)`` softmax rows.
        class_index:   The class the chosen image must be predicted as.
        threshold:     Minimum confidence (strictly greater than).

    Returns:
        ``(row_index, confidence)``.

    Raises:
        ValueError: if no image is predicted *class_index* above *threshold*.
    """
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2:
        raise ValueError(f"expected a 2-D (N, C) array, got shape {probs.shape}.")
    predicted = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    eligible = (predicted == class_index) & (confidence > threshold)
    if not eligible.any():
        matching = confidence[predicted == class_index]
        best_seen = float(matching.max()) if matching.size else 0.0
        raise ValueError(
            f"No image predicted class {class_index} above {threshold:.2f} "
            f"(best matching confidence was {best_seen:.3f})."
        )
    candidates = np.where(eligible)[0]
    best = candidates[np.argmax(confidence[candidates])]
    return int(best), float(confidence[best])


def probabilities_report(
    probabilities: NDArray, labels: Sequence[str] = FER_EMOTIONS
) -> Dict[str, float]:
    """A ``{label: probability}`` dict for one image's softmax vector (for saving).

    Raises:
        ValueError: if the vector length does not match the label count.
    """
    vector = np.asarray(probabilities, dtype=float).reshape(-1)
    if vector.shape[0] != len(labels):
        raise ValueError(
            f"expected {len(labels)} probabilities, got {vector.shape[0]}."
        )
    return {label: float(p) for label, p in zip(labels, vector)}
