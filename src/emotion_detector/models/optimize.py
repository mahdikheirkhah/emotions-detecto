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

import time
from typing import Any, Callable, Dict, Optional

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
