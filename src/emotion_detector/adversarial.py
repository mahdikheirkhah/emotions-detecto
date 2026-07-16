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

from dataclasses import dataclass
from typing import Any, Callable, Dict, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from src.emotion_detector.models.labels import FER_EMOTIONS

# grad_fn(x, target_index) -> d(loss)/d(x) (same shape as x); predict_fn(x) -> softmax.
GradFn = Callable[[NDArray, int], NDArray]
PredictFn = Callable[[NDArray], NDArray]


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


# ===========================================================================
# The attack (#58): FGSM / iterative-FGSM (BIM) toward a target class.
# ===========================================================================


@dataclass(frozen=True)
class AttackResult:
    """Outcome of an attack: the adversarial image, its perturbation, and both preds."""

    adversarial: NDArray  # perturbed input, same shape as x, clipped to [0, 1]
    perturbation: NDArray  # adversarial - original (its L-inf norm is <= epsilon)
    iterations: int  # steps actually taken (1 for FGSM; <= `iterations` for BIM)
    success: bool  # did argmax(adversarial) become the target class?
    original_probs: NDArray  # softmax before the attack
    adversarial_probs: NDArray  # softmax after the attack


def fgsm_perturbation(gradient: NDArray, epsilon: float) -> NDArray:
    """One signed FGSM step: ``epsilon * sign(gradient)`` (pure, no TF).

    ``sign`` makes the step a fixed ``epsilon`` per pixel, so its L-infinity norm is
    exactly ``epsilon`` — the imperceptibility budget.
    """
    return float(epsilon) * np.sign(np.asarray(gradient, dtype=float))


def run_attack(
    x: NDArray,
    target_index: int,
    grad_fn: GradFn,
    predict_fn: PredictFn,
    epsilon: float,
    attack_type: str = "fgsm",
    iterations: int = 10,
    step_size: float | None = None,
) -> AttackResult:
    """Perturb *x* toward *target_index* until it flips (or the budget runs out).

    **Targeted:** ``grad_fn`` returns ``d(loss)/d(x)`` for cross-entropy toward the
    target, so to *raise* the target probability we step **against** it
    (``x - sign(grad)``). The gradient is w.r.t. the pixels, model frozen -- the point.

    * ``fgsm``: one step of size ``epsilon`` -- ``x - epsilon * sign(grad)``.
    * ``bim`` (iterative FGSM): up to ``iterations`` small ``step_size`` steps (default
      ``epsilon / iterations``), each re-clipped into the ``epsilon`` L-inf ball and
      into ``[0, 1]``, stopping early once it flips: subtler than one big FGSM jump.

    Args:
        x: The source input, e.g. ``(48, 48, 1)`` normalized to ``[0, 1]``.
        target_index: The class to flip the prediction *to*.
        grad_fn / predict_fn: Injected model hooks (real Keras via
            ``keras_attack_functions`` in production; fakes in tests).
        epsilon: L-infinity perturbation budget.
        attack_type: ``"fgsm"`` | ``"bim"``.
        iterations / step_size: BIM only.

    Returns:
        An ``AttackResult``.

    Raises:
        ValueError: on an unknown ``attack_type`` or ``epsilon <= 0``.
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}.")
    original = np.asarray(x, dtype=float)
    original_probs = np.asarray(predict_fn(original)).reshape(-1)

    if attack_type == "fgsm":
        adversarial = np.clip(
            original - fgsm_perturbation(grad_fn(original, target_index), epsilon),
            0.0,
            1.0,
        )
        taken = 1
    elif attack_type == "bim":
        alpha = float(step_size) if step_size else epsilon / max(int(iterations), 1)
        adversarial = original.copy()
        taken = 0
        for step in range(int(iterations)):
            taken = step + 1
            adversarial = adversarial - alpha * np.sign(
                grad_fn(adversarial, target_index)
            )
            adversarial = np.clip(adversarial, original - epsilon, original + epsilon)
            adversarial = np.clip(adversarial, 0.0, 1.0)
            if int(np.argmax(predict_fn(adversarial))) == target_index:
                break
    else:
        raise ValueError(f"unknown attack_type '{attack_type}' (use 'fgsm' or 'bim').")

    adversarial_probs = np.asarray(predict_fn(adversarial)).reshape(-1)
    return AttackResult(
        adversarial=adversarial,
        perturbation=adversarial - original,
        iterations=taken,
        success=int(np.argmax(adversarial_probs)) == target_index,
        original_probs=original_probs,
        adversarial_probs=adversarial_probs,
    )


def keras_attack_functions(model: Any) -> Tuple[GradFn, PredictFn]:
    """Build ``(grad_fn, predict_fn)`` for a Keras model (``tf.GradientTape``, lazy TF).

    ``grad_fn`` records the forward pass on a watched input tensor and returns
    ``d(cross-entropy toward target)/d(input)`` by reverse-mode autodiff -- the input
    gradient with the trained weights fixed. ``predict_fn`` is a plain forward pass.
    """
    import tensorflow as tf
    from tensorflow.keras.losses import CategoricalCrossentropy

    cce = CategoricalCrossentropy()
    num_classes = int(model.output_shape[-1])

    def grad_fn(x: NDArray, class_index: int) -> NDArray:
        x_t = tf.convert_to_tensor(np.asarray(x, dtype=np.float32)[np.newaxis, ...])
        target = tf.one_hot([int(class_index)], num_classes)
        with tf.GradientTape() as tape:
            tape.watch(x_t)
            prediction = model(x_t, training=False)
            loss = cce(target, prediction)
        gradient = tape.gradient(loss, x_t)
        return gradient.numpy()[0]

    def predict_fn(x: NDArray) -> NDArray:
        x_t = np.asarray(x, dtype=np.float32)[np.newaxis, ...]
        return np.asarray(model(x_t, training=False)).reshape(-1)

    return grad_fn, predict_fn
