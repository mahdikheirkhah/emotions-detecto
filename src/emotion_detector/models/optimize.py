"""Optional post-training quantization → a smaller, faster TFLite model.

**Quantization** stores weights/activations in lower precision (float32 → float16 or
int8), shrinking the model and speeding up inference — the difference between a webcam
loop that hits "1 prediction/second" on modest hardware and one that stutters.

This is **post-training** quantization: a quick post-hoc conversion of an
already-trained model (vs *quantization-aware* training, which simulates quantization
during training for less accuracy loss but must be built into the training loop).

Modes (config ``optimization.quantization``), cheapest-accuracy-loss first:
  * ``none``    — float32 TFLite (format only; a baseline for fair comparison).
  * ``float16`` — half-precision weights (~2× smaller, tiny accuracy loss).
  * ``dynamic`` — dynamic-range int8 *weights*, float activations (~4× smaller).
  * ``int8``    — full int8 (weights **and** activations); ~4× smaller and fastest, but
                  needs a **representative dataset** to calibrate the float→int ranges.

Nothing is shipped without measuring the cost: ``optimize_model`` reports size, latency
**and** accuracy before/after so the accuracy/latency trade-off is explicit
(CONTRIBUTING §8). It is just another config toggle — full vs quantized compared fairly
(Ablation §3). TensorFlow is imported lazily so importing this module stays cheap.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from src.emotion_detector.utils.logging import logger

_MODES = ("none", "dynamic", "float16", "int8")


def _representative_dataset(data: NDArray) -> Callable:
    """Yield one float32 sample at a time for int8 range calibration."""

    def generator():
        for sample in data:
            yield [sample[None].astype("float32")]  # (1, H, W, C)

    return generator


def quantize(
    model: Any,
    cfg: dict,
    representative_data: Optional[NDArray] = None,
) -> bytes:
    """Convert *model* to a TFLite flatbuffer using ``optimization.quantization``.

    Args:
        model: A trained Keras model.
        cfg:   Loaded config (reads ``optimization.quantization``).
        representative_data: ``(N, H, W, C)`` calibration samples — **required for
            ``int8``** (used to measure the float ranges the ints must cover).

    Returns:
        The serialized ``.tflite`` model as ``bytes``.

    Raises:
        ValueError: on an unknown mode, or ``int8`` without representative data.
    """
    import tensorflow as tf

    mode = cfg["optimization"]["quantization"]
    if mode not in _MODES:
        raise ValueError(
            f"Unknown optimization.quantization '{mode}'. Valid: {', '.join(_MODES)}."
        )

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if mode == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif mode == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif mode == "int8":
        if representative_data is None or len(representative_data) == 0:
            raise ValueError(
                "int8 quantization needs a representative dataset to calibrate ranges."
            )
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = _representative_dataset(representative_data)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    # mode == "none": no optimizations → plain float32 TFLite.

    tflite_bytes = converter.convert()
    logger.info(f"Quantized ({mode}) → {len(tflite_bytes):,} bytes TFLite.")
    return tflite_bytes


class TFLitePredictor:
    """Run inference with a TFLite model, hiding the int8 (de)quantization.

    For an int8 model the input/output tensors are integers with a ``(scale,
    zero_point)``; this wrapper quantizes the incoming float image and dequantizes the
    logits back to float, so callers pass/receive float arrays regardless of mode.
    """

    def __init__(self, tflite_bytes: bytes) -> None:
        import tensorflow as tf

        self._it = tf.lite.Interpreter(model_content=tflite_bytes)
        self._it.allocate_tensors()
        self._inp = self._it.get_input_details()[0]
        self._out = self._it.get_output_details()[0]

    def _quantize_input(self, x: NDArray) -> NDArray:
        dtype = self._inp["dtype"]
        if dtype in (np.int8, np.uint8):
            scale, zero_point = self._inp["quantization"]
            x = np.round(x / scale + zero_point)
            info = np.iinfo(dtype)
            x = np.clip(x, info.min, info.max)
        return x.astype(dtype)

    def _dequantize_output(self, y: NDArray) -> NDArray:
        if self._out["dtype"] in (np.int8, np.uint8):
            scale, zero_point = self._out["quantization"]
            y = (y.astype(np.float32) - zero_point) * scale
        return y.astype(np.float32)

    def predict(self, X: NDArray) -> NDArray:
        """Softmax scores ``(N, num_classes)`` for a float batch ``(N, H, W, C)``."""
        X = np.asarray(X, dtype="float32")
        out = []
        for x in X:
            xb = self._quantize_input(x[None])  # tflite default batch = 1
            self._it.set_tensor(self._inp["index"], xb)
            self._it.invoke()
            y = self._it.get_tensor(self._out["index"])
            out.append(self._dequantize_output(y)[0])
        return np.array(out)

    def predict_classes(self, X: NDArray) -> NDArray:
        """Argmax-decoded class indices ``(N,)``."""
        return np.argmax(self.predict(X), axis=1)


def _accuracy(y_pred: NDArray, y_true: NDArray) -> float:
    y_true = np.asarray(y_true)
    if y_true.ndim > 1:  # one-hot → indices
        y_true = np.argmax(y_true, axis=1)
    return float(np.mean(y_pred == y_true))


def _latency_ms(predict_once: Callable[[], Any], runs: int = 50) -> float:
    """Mean per-call latency in ms (single-image), after one warm-up call."""
    predict_once()  # warm up (graph build / XNNPACK)
    start = time.perf_counter()
    for _ in range(runs):
        predict_once()
    return (time.perf_counter() - start) / runs * 1000.0


def optimize_model(
    model: Any,
    cfg: dict,
    X_eval: NDArray,
    y_eval: NDArray,
    representative_data: Optional[NDArray] = None,
) -> Dict[str, Any]:
    """Quantize *model*, measure size/latency/accuracy before vs after, log the deltas.

    The int8 path uses ``representative_data`` (falls back to ``X_eval``) for
    calibration. Returns a report dict; ``passed`` is False if the accuracy drop
    exceeds ``optimization.max_accuracy_drop`` (logged as a warning — never silently
    ship a lossy model).
    """
    opt = cfg["optimization"]
    mode = opt["quantization"]
    num_classes = cfg["model"]["num_classes"]

    X_eval = np.asarray(X_eval, dtype="float32")
    rep = representative_data if representative_data is not None else X_eval

    tflite_bytes = quantize(model, cfg, representative_data=rep)
    predictor = TFLitePredictor(tflite_bytes)

    # --- size: float32 weights (params × 4 bytes) vs the TFLite flatbuffer ---
    keras_size = int(model.count_params() * 4)
    tflite_size = len(tflite_bytes)

    # --- accuracy: full-precision Keras vs quantized TFLite, same eval set ---
    keras_pred = np.argmax(model.predict(X_eval, verbose=0), axis=1)
    tflite_pred = predictor.predict_classes(X_eval)
    keras_acc = _accuracy(keras_pred, y_eval)
    tflite_acc = _accuracy(tflite_pred, y_eval)
    drop = keras_acc - tflite_acc

    # --- latency: single-image inference (the webcam's per-frame cost) ---
    one = X_eval[:1]
    keras_ms = _latency_ms(lambda: model(one, training=False))
    tflite_ms = _latency_ms(lambda: predictor.predict(one))

    passed = drop <= opt.get("max_accuracy_drop", 1.0)
    report = {
        "mode": mode,
        "num_classes": num_classes,
        "keras_size_bytes": keras_size,
        "tflite_size_bytes": tflite_size,
        "size_reduction": round(keras_size / tflite_size, 2) if tflite_size else 0.0,
        "keras_accuracy": round(keras_acc, 4),
        "tflite_accuracy": round(tflite_acc, 4),
        "accuracy_drop": round(drop, 4),
        "keras_latency_ms": round(keras_ms, 3),
        "tflite_latency_ms": round(tflite_ms, 3),
        "speedup": round(keras_ms / tflite_ms, 2) if tflite_ms else 0.0,
        "passed": passed,
        "tflite_bytes": tflite_bytes,
    }

    logger.info(
        f"Quantization '{mode}': size {keras_size:,}→{tflite_size:,} B "
        f"({report['size_reduction']}×), accuracy {keras_acc:.4f}→{tflite_acc:.4f} "
        f"(drop {drop:+.4f}), latency {keras_ms:.2f}→{tflite_ms:.2f} ms "
        f"({report['speedup']}× faster)."
    )
    if not passed:
        logger.warning(
            f"Accuracy drop {drop:.4f} exceeds max_accuracy_drop "
            f"{opt.get('max_accuracy_drop')} — review before shipping this model."
        )
    return report


def save_tflite(tflite_bytes: bytes, path: str) -> None:
    """Write the TFLite flatbuffer to *path* (creating parent dirs)."""
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(tflite_bytes)
    logger.info(f"Saved TFLite model → {out} ({len(tflite_bytes):,} bytes)")


# ===========================================================================
# Weight pruning (#48) — magnitude-based, Keras-3 native
# ===========================================================================
# **Pruning** zeroes the least-important weights to make the model *sparse*. We prune
# by **magnitude** — the smallest-|w| weights are the safest to drop, since they
# contribute least to the output — then **fine-tune** so the surviving weights
# compensate for the cut connections (the prune→fine-tune cycle). Sparsity shrinks the
# *compressed* model (zeros gzip away) and can speed up inference *if* the runtime has
# sparse kernels; on a dense CPU/GPU kernel the win is mostly size, so it pairs best
# with quantization (#47).
#
# NOTE — why not ``tensorflow_model_optimization``: the issue names tfmot, but tfmot
# 0.8.x is built on Keras 2 and **crashes on Keras 3** (which TF 2.21 uses) —
# ``prune_low_magnitude`` fails cloning a Keras-3 model ("The added layer must be an
# instance of class Layer"). Using it would force the whole project onto legacy Keras 2.
# So this is a faithful **reimplementation** of tfmot's approach on the project's stack:
# ``prune`` == ``prune_low_magnitude`` + fine-tune + ``strip_pruning`` (the exported
# model is already a plain sparse Keras model — no wrappers to strip), the pruning
# callback == ``UpdatePruningStep``; the schedule == ``PolynomialDecay``.


def _prunable_layers(model: Any) -> List[Any]:
    """Layers with a trainable ``kernel`` (Conv2D / Dense); biases/BN untouched."""
    return [layer for layer in model.layers if hasattr(layer, "kernel")]


def polynomial_decay_sparsity(
    step: int,
    total_steps: int,
    initial_sparsity: float = 0.0,
    final_sparsity: float = 0.5,
    power: float = 3.0,
) -> float:
    """Sparsity target at *step*, ramping ``initial → final`` (tfmot PolynomialDecay).

    Starts gentle (few weights cut) and raises the threshold over training so the model
    adapts gradually instead of losing many connections at once: ``final + (initial -
    final) * (1 - step/total)^power``.
    """
    if total_steps <= 0:
        return final_sparsity
    frac = min(max(step / total_steps, 0.0), 1.0)
    return final_sparsity + (initial_sparsity - final_sparsity) * (
        (1.0 - frac) ** power
    )


def _apply_layer_sparsity(layer: Any, sparsity_frac: float) -> None:
    """Zero the smallest-magnitude ``sparsity_frac`` of *layer*'s kernel, in place."""
    if sparsity_frac <= 0:
        return
    w = np.asarray(layer.kernel.numpy())
    flat = np.abs(w).ravel()
    k = int(round(sparsity_frac * flat.size))
    if k <= 0:
        return
    if k >= flat.size:
        layer.kernel.assign(np.zeros_like(w))
        return
    drop_idx = np.argpartition(flat, k - 1)[:k]  # indices of the k smallest |w|
    flat_w = w.ravel().copy()
    flat_w[drop_idx] = 0.0
    layer.kernel.assign(flat_w.reshape(w.shape).astype(w.dtype))


def apply_pruning(model: Any, sparsity_frac: float) -> None:
    """One-shot magnitude-prune each prunable layer to *sparsity_frac* (in place)."""
    for layer in _prunable_layers(model):
        _apply_layer_sparsity(layer, sparsity_frac)


def sparsity(model: Any) -> float:
    """Fraction of kernel weights that are exactly zero (across all prunable layers)."""
    total = zeros = 0
    for layer in _prunable_layers(model):
        w = np.asarray(layer.kernel.numpy())
        total += w.size
        zeros += int(np.sum(w == 0))
    return zeros / total if total else 0.0


def _make_pruning_callback(
    final_sparsity: float, total_steps: int, frequency: int
) -> Any:
    """A Keras callback (tfmot ``UpdatePruningStep``) re-masking on the schedule.

    Every ``frequency`` steps it raises the sparsity target (polynomial decay) and
    re-zeroes the smallest weights; a final pass at ``on_train_end`` pins the exact
    target so pruned connections stay cut after fine-tuning.
    """
    from tensorflow import keras

    class _PruningCallback(keras.callbacks.Callback):
        def __init__(self) -> None:
            super().__init__()
            self._step = 0

        def on_train_batch_end(self, batch, logs=None) -> None:
            self._step += 1
            if self._step % frequency == 0:
                target = polynomial_decay_sparsity(
                    self._step, total_steps, 0.0, final_sparsity
                )
                for layer in _prunable_layers(self.model):
                    _apply_layer_sparsity(layer, target)

        def on_train_end(self, logs=None) -> None:
            for layer in _prunable_layers(self.model):
                _apply_layer_sparsity(layer, final_sparsity)

    return _PruningCallback()


def prune(
    model: Any,
    cfg: dict,
    X: Optional[NDArray] = None,
    y: Optional[NDArray] = None,
) -> Any:
    """Magnitude-prune *model* to ``optimization.pruning.target_sparsity``, in place.

    With fine-tune data (``X``, one-hot ``y``) it runs the **prune → fine-tune** cycle:
    the callback ramps sparsity over ``fine_tune_epochs`` while training lets the
    surviving weights recover accuracy. Without data it prunes one-shot (no recovery).
    Returns the same (now sparse) model — ready to save/quantize, no wrappers to strip.

    Raises:
        ValueError: if ``target_sparsity`` is not in ``[0, 1)``.
    """
    p = cfg["optimization"]["pruning"]
    target = float(p["target_sparsity"])
    if not 0.0 <= target < 1.0:
        raise ValueError(f"target_sparsity must be in [0, 1), got {target}.")

    if X is None or y is None:
        apply_pruning(model, target)
        logger.info(f"Pruned one-shot (no fine-tune) → sparsity {sparsity(model):.3f}.")
        return model

    epochs = int(p.get("fine_tune_epochs", 2))
    batch = int(cfg["model"].get("batch_size", 64))
    frequency = int(p.get("frequency", 100))
    total_steps = max(1, math.ceil(len(X) / batch)) * epochs

    logger.info(f"Pruning → target sparsity {target} over {epochs} fine-tune epoch(s)…")
    model.fit(
        X,
        y,
        epochs=epochs,
        batch_size=batch,
        verbose=0,
        callbacks=[_make_pruning_callback(target, total_steps, frequency)],
    )
    apply_pruning(model, target)  # guarantee the exact final sparsity
    logger.info(f"Pruned + fine-tuned → sparsity {sparsity(model):.3f}.")
    return model


def _gzipped_kernel_bytes(model: Any) -> int:
    """Compressed size of the kernel weights — where pruning's size win actually shows.

    Raw float32 storage does not shrink when weights become zero; the benefit appears
    under compression (the zeros collapse) or when paired with quantization (#47).
    """
    import gzip

    raw = b"".join(
        np.asarray(layer.kernel.numpy(), dtype=np.float32).tobytes()
        for layer in _prunable_layers(model)
    )
    return len(gzip.compress(raw))


def prune_and_report(
    model: Any,
    cfg: dict,
    X_eval: NDArray,
    y_eval: NDArray,
    X_ft: Optional[NDArray] = None,
    y_ft: Optional[NDArray] = None,
) -> Dict[str, Any]:
    """Prune *model* and report dense-vs-pruned sparsity, compressed size, accuracy.

    Measures accuracy on the same ``X_eval`` before and after so the sparsity/accuracy
    trade-off is explicit; ``passed`` is False if the drop exceeds
    ``optimization.max_accuracy_drop`` (CONTRIBUTING §8). Prunes in place.
    """
    X_eval = np.asarray(X_eval, dtype="float32")

    dense_acc = _accuracy(np.argmax(model.predict(X_eval, verbose=0), axis=1), y_eval)
    dense_gzip = _gzipped_kernel_bytes(model)

    prune(model, cfg, X_ft, y_ft)

    pruned_acc = _accuracy(np.argmax(model.predict(X_eval, verbose=0), axis=1), y_eval)
    pruned_gzip = _gzipped_kernel_bytes(model)
    drop = dense_acc - pruned_acc
    passed = drop <= cfg["optimization"].get("max_accuracy_drop", 1.0)

    report = {
        "target_sparsity": float(cfg["optimization"]["pruning"]["target_sparsity"]),
        "achieved_sparsity": round(sparsity(model), 4),
        "dense_gzip_bytes": dense_gzip,
        "pruned_gzip_bytes": pruned_gzip,
        "gzip_reduction": round(dense_gzip / pruned_gzip, 2) if pruned_gzip else 0.0,
        "dense_accuracy": round(dense_acc, 4),
        "pruned_accuracy": round(pruned_acc, 4),
        "accuracy_drop": round(drop, 4),
        "passed": passed,
    }
    logger.info(
        f"Pruning: sparsity {report['achieved_sparsity']}, gzip "
        f"{dense_gzip:,}→{pruned_gzip:,} B ({report['gzip_reduction']}×), accuracy "
        f"{dense_acc:.4f}→{pruned_acc:.4f} (drop {drop:+.4f})."
    )
    if not passed:
        logger.warning(
            f"Pruned accuracy drop {drop:.4f} exceeds max_accuracy_drop "
            f"{cfg['optimization'].get('max_accuracy_drop')} — review before shipping."
        )
    return report
