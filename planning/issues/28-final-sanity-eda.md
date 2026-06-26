---
title: "[Phase 3] Final sanity-check EDA on model-ready tensors"
labels: ["phase-3-features", "notebook"]
---
## 1. Description

One last look before modeling: inspect the exact tensors that will be fed to the network. Confirm shapes, dtypes, value ranges, one-hot correctness, and that a few decoded samples still look like the right faces. This catches pipeline bugs that are invisible until they silently wreck training.

## 2. Learning Objective

- **"Garbage in, garbage out":** the final guard against a subtle preprocessing bug.
- **Tensor literacy:** reading `(batch, height, width, channels)` and label one-hot rows fluently.
- **Round-trip verification:** decode a model-ready tensor back to an image and confirm the label.
- **Range/dtype discipline:** floats in the expected range, labels summing to 1.

## 3. To-Do list for coding

- [ ] Pull one batch from the training `tf.data.Dataset`
- [ ] Assert shapes `(B,48,48,1)` / `(B,7)`, dtype, value range; one-hot rows sum to 1
- [ ] Decode and `imshow` a few samples with their argmax label name
- [ ] Confirm train batches are shuffled and (if on) augmented; val/test are not

## 4. Code learning (packages & methods)

- **`tensorflow`** — iterate a `Dataset` (`take(1)`), `numpy()` conversion
- **`numpy`** — `argmax`, range/sum assertions
- **`matplotlib`** — `imshow` decoded samples

➡️ **After we implement:** you explain what each assertion guarantees about the inputs. I'll explain why even a correct-looking accuracy can hide a label/preprocessing misalignment that only this round-trip catches.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — reproducible inspection, assertions over eyeballing.

> 🔀 **Note — Ablation-Driven Architecture:** Run this check for each stage configuration you plan to train, so you trust the inputs before attributing accuracy changes to a toggle. See `CONTRIBUTING.md` §3.
