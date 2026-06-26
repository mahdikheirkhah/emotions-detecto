---
title: "[Phase 3] Reshape + one-hot encode + build tf.data pipeline"
labels: ["phase-3-features", "data"]
---
## 1. Description

Turn the cleaned, split arrays into exactly what Keras expects: images shaped `(N, 48, 48, 1)`, labels one-hot encoded to `(N, 7)`, wrapped in efficient `tf.data.Dataset` pipelines (batch, shuffle on train, prefetch). Augmentation (#22) plugs into the training dataset here.

## 2. Learning Objective

- **The channel dimension:** why Conv2D needs `(H, W, C)` and grayscale is `C=1`.
- **One-hot vs integer labels:** matching the loss (`categorical_crossentropy` vs `sparse_categorical_crossentropy`).
- **`tf.data` performance:** how `shuffle`, `batch`, `prefetch`, and `cache` keep the GPU fed.
- **Where augmentation belongs:** applied to the training dataset only, after batching.

## 3. To-Do list for coding

- [ ] `data/pipeline.py` → `to_tensors(X, y) -> (images[N,48,48,1], onehot[N,7])`
- [ ] `make_dataset(X, y, cfg, training: bool) -> tf.data.Dataset` (shuffle+augment only when training)
- [ ] Read `batch_size`, `shuffle_buffer` from config
- [ ] `tests/test_pipeline.py`: shapes `(B,48,48,1)`/`(B,7)`; train set is shuffled; val/test are not

## 4. Code learning (packages & methods)

- **`numpy`** — `reshape(-1, 48, 48, 1)`, `expand_dims`
- **`tensorflow.keras.utils`** — `to_categorical`
- **`tensorflow.data`** — `Dataset.from_tensor_slices`, `shuffle`, `batch`, `map`, `prefetch`, `cache`

➡️ **After we implement:** you explain why grayscale still needs a channel axis and how one-hot pairs with the loss. I'll explain how `tf.data` builds a streaming input graph and how `prefetch` overlaps data prep with model compute.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — type hints, docstrings, shape-assert tests.

> 🔀 **Note — Ablation-Driven Architecture:** `batch_size`/`shuffle_buffer` live in `config.yaml`; the augmentation map respects its stage toggle. See `CONTRIBUTING.md` §3.
