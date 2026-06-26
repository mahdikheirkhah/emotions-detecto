---
title: "[Phase 4] Build-b — assemble the full CNN (config-driven ModelBuilder)"
labels: ["phase-4-model", "model"]
---
## 1. Description

Assemble the complete from-scratch CNN by stacking conv blocks (#33) into a classifier head, behind a `BaseModelBuilder`. The chosen architecture (`simple_cnn` / `vgg_small` / `resnet_mini`) is selected by dispatch from `config.yaml`, so swapping architectures is a one-line config change.

## 2. Learning Objective

- **From blocks to a network:** how stacked blocks + a dense head form an end-to-end classifier.
- **The classification head:** `GlobalAveragePooling`/`Flatten → Dense → Dropout → Dense(7, softmax)`.
- **The Strategy pattern for architectures:** each architecture is a builder selected by config.
- **`model.summary()`:** reading layer shapes and the total parameter budget.

## 3. To-Do list for coding

- [ ] `models/builders.py` → `SimpleCnnBuilder`, `VggSmallBuilder` (both `BaseModelBuilder`), each `build(input_shape, n_classes) -> keras.Model`
- [ ] `build_model(cfg)` dispatch on `model.architecture`
- [ ] Output `Dense(7, activation="softmax")`; print `model.summary()`
- [ ] `tests/test_builders.py`: output shape `(None, 7)`; unknown architecture raises `ValueError`

## 4. Code learning (packages & methods)

- **`tensorflow.keras`** — `Input`, `Model`, `layers.Flatten`/`GlobalAveragePooling2D`, `Dense`, `Dropout`, `model.summary()`
- **`abc`** — implements `BaseModelBuilder`

➡️ **After we implement:** you walk me through `model.summary()` and explain where the parameters concentrate. I'll explain how Keras builds the layer graph and infers each layer's output shape symbolically before any data flows.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) §4 — `BaseModelBuilder` subclasses, dispatch, tests.

> 🔀 **Note — Ablation-Driven Architecture:** `model.architecture` selects the builder; add new architectures as new options + branches, never by deleting old ones. See `CONTRIBUTING.md` §3.
