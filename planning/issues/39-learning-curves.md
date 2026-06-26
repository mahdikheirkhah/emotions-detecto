---
title: "[Phase 4] Learning-curves plot (learning_curves.png) + validation_loss_accuracy.py"
labels: ["phase-4-model", "model"]
---
## 1. Description

Produce the `results/model/learning_curves.png` plot (train vs validation loss & accuracy over epochs) and the `scripts/validation_loss_accuracy.py` helper. The plot must visibly show training stopping **before** validation loss diverges — exactly what the audit checks.

## 2. Learning Objective

- **What learning curves diagnose:** under-fitting (both high), good fit (both low, close), over-fitting (train ↓ while val ↑).
- **The overfitting signature:** the epoch where val loss turns up while train keeps falling.
- **Why we plot loss *and* accuracy:** they can tell slightly different stories.
- **Connecting to early stopping:** the chosen stop epoch should sit at/just past the val-loss minimum.

## 3. To-Do list for coding

- [ ] `scripts/validation_loss_accuracy.py` → load saved `history`, plot train/val loss and accuracy
- [ ] Mark the early-stopping epoch; ensure the curves show the pre-overfit stop
- [ ] Save `results/model/learning_curves.png`
- [ ] `tests/test_curves.py`: given a fake history, a PNG file is produced

## 4. Code learning (packages & methods)

- **`matplotlib`** — `plot`, `twin`/subplots for loss vs accuracy, `axvline` for the stop epoch, `savefig`
- **`json`** — load the persisted history

➡️ **After we implement:** you read the curves and tell me where overfitting would begin and why early stopping fired when it did. I'll explain how to distinguish noise from a genuine divergence trend in the validation curve.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) §8 — the curves are the evidence that training stopped before overfitting.

> 🔀 **Note — Ablation-Driven Architecture:** Regenerate curves per ablation run (e.g. augmentation on vs off) to *see* each toggle's effect on the overfitting gap. See `CONTRIBUTING.md` §3.
