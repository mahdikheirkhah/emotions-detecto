---
title: "[Phase 6] Model optimization — quantization (optional, toggleable)"
labels: ["phase-6-optional", "optional", "model"]
---
## 1. Description

**Optional:** quantize the trained model (e.g. post-training float16/int8 via TFLite) to shrink it and speed up inference for the real-time webcam loop, while checking the accuracy drop stays acceptable. Toggleable via config.

## 2. Learning Objective

- **What quantization is:** representing weights/activations in lower precision (float32 → int8/float16) to cut size and latency.
- **Post-training vs quantization-aware:** quick post-hoc conversion vs simulating quantization during training for less accuracy loss.
- **The accuracy/latency trade-off:** measuring both before claiming a win.
- **Why it matters for "1 prediction/second" on a webcam** on modest hardware.

## 3. To-Do list for coding

- [ ] `models/optimize.py` → `quantize(model, cfg)` using `tf.lite.TFLiteConverter` (dynamic-range / float16 / int8)
- [ ] Config: `optimization.quantization: none | float16 | int8` + toggle
- [ ] Measure size + inference latency + accuracy before/after; log the deltas
- [ ] Save the quantized artifact; `tests/test_optimize.py`: quantized model still predicts a valid class

## 4. Code learning (packages & methods)

- **`tensorflow.lite`** — `TFLiteConverter.from_keras_model`, `Optimize.DEFAULT`, `target_spec`
- **`time`** — latency measurement
- **`numpy`** — accuracy comparison

➡️ **After we implement:** you report the size/latency/accuracy trade-off and whether it's worth it. I'll explain how int8 quantization maps float ranges to integers via per-tensor scale + zero-point and where the accuracy loss comes from.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — optional, measured, documented; never ship a quantized model without the accuracy check.

> 🔀 **Note — Ablation-Driven Architecture:** `optimization.quantization` is a config toggle; compare full vs quantized inference fairly. See `CONTRIBUTING.md` §3.
