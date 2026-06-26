---
title: "[Phase 3] FE-c — data augmentation (rotation / flip / zoom / shift) (toggleable)"
labels: ["phase-3-features", "data"]
---
## 1. Description

Add training-time **data augmentation** as a toggleable strategy: random horizontal flip, small rotations, zoom, and shifts. Augmentation synthesizes plausible variations of faces, enlarging the effective training set and directly fighting overfitting (a core project requirement).

## 2. Learning Objective

- **Why augmentation regularizes:** showing the model many transformed views teaches invariance (a smile is a smile, flipped or slightly rotated) and reduces overfitting.
- **Label-preserving transforms only:** horizontal flip is fine for faces; vertical flip or huge rotations are not (they create unnatural faces).
- **Train-only:** augmentation applies to training batches, never to validation/test.
- **On-the-fly vs precomputed:** generating augmented batches during training vs materializing them.

## 3. To-Do list for coding

- [ ] `data/augmentation.py` → `build_augmenter(cfg)` returning a Keras augmentation pipeline or `ImageDataGenerator`
- [ ] Read params from `config.yaml` (`augmentation.rotation`, `flip`, `zoom`, `shift`)
- [ ] Gate on the `augmentation` stage toggle (off → identity)
- [ ] `tests/test_augmentation.py`: augmented batch keeps shape/labels; stage-off returns inputs unchanged

## 4. Code learning (packages & methods)

- **`tensorflow.keras.layers`** — `RandomFlip`, `RandomRotation`, `RandomZoom`, `RandomTranslation` (preprocessing layers)
- *(or)* **`keras.preprocessing.image.ImageDataGenerator`** — classic generator API
- **`numpy`** — sanity checks in tests

➡️ **After we implement:** you explain why we only use label-preserving transforms and keep augmentation off the test set. I'll explain how Keras preprocessing layers apply random transforms per-batch on the GPU and why that's cheaper than precomputing.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — seeded randomness, train-only, tests incl. stage-off.

> 🔀 **Note — Ablation-Driven Architecture:** `augmentation` is its own stage toggle — turning it off isolates how much it contributes to the overfitting gap. See `CONTRIBUTING.md` §3.
