---
title: "[Phase 4] Callbacks — EarlyStopping + ModelCheckpoint + ReduceLROnPlateau"
labels: ["phase-4-model", "model"]
---
## 1. Description

Build the training callbacks the subject explicitly asks for: **early stopping** (stop before overfitting), **model checkpointing** (keep the best weights), and **LR reduction on plateau**. All thresholds/patience come from `config.yaml`. These are the mechanisms that make the "not overfitting" audit pass.

## 2. Learning Objective

- **Early stopping as regularization:** monitoring validation loss and halting when it stops improving — *why* this prevents overfitting.
- **`patience` and `restore_best_weights`:** tolerating noisy epochs while keeping the best model.
- **Checkpointing:** persisting the best model so a crash/late-epoch overfit doesn't lose it.
- **LR scheduling:** why dropping the learning rate on a plateau can unlock further improvement.

## 3. To-Do list for coding

- [ ] `models/callbacks.py` → `build_callbacks(cfg) -> list`
- [ ] `EarlyStopping(monitor="val_loss", patience=..., restore_best_weights=True)`
- [ ] `ModelCheckpoint(save_best_only=True)` → `results/model/final_emotion_model.keras`
- [ ] `ReduceLROnPlateau(monitor="val_loss", factor=..., patience=...)`
- [ ] (TensorBoard callback added in #37) ; `tests/test_callbacks.py`: returns configured callbacks

## 4. Code learning (packages & methods)

- **`tensorflow.keras.callbacks`** — `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`

➡️ **After we implement:** you explain how early stopping decides to halt and what `restore_best_weights` does. I'll explain how Keras invokes callbacks at each `on_epoch_end` hook and how `EarlyStopping` tracks the best monitored value + wait counter.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) §8 — early stopping + checkpointing are mandatory anti-overfitting tools.

> 🔀 **Note — Ablation-Driven Architecture:** `patience`, `factor`, monitored metric are config values; you can ablate early stopping itself to *see* the overfitting it prevents. See `CONTRIBUTING.md` §3.
