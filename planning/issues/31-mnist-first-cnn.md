---
title: "[Phase 3.5] MNIST — first CNN (Keras) + compare to baseline"
labels: ["phase-3.5-preliminary", "model"]
---
## 1. Description

Build and train your **first CNN** — a small Keras conv net on MNIST — and compare it head-to-head with the logistic-regression baseline (#30). This is where convolutions "click" before we tackle the harder emotion task, and it establishes the Keras training loop we'll reuse.

## 2. Learning Objective

- **What a convolution computes:** a small learnable kernel sliding over the image, producing feature maps that detect edges/strokes.
- **The CNN trio:** convolution (feature extraction) → pooling (downsampling/invariance) → dense head (classification).
- **Why a CNN beats logistic regression on images:** parameter sharing + locality exploit spatial structure the linear model ignored.
- **The Keras `Sequential` → `compile` → `fit` → `evaluate` loop** we'll scale up in Phase 4.

## 3. To-Do list for coding

- [ ] Build a small CNN: `Conv2D → ReLU → MaxPool → Conv2D → ReLU → MaxPool → Flatten → Dense → softmax`
- [ ] `compile` (Adam, categorical-crossentropy, accuracy); `fit` with validation
- [ ] Evaluate on test; compare accuracy to the #30 baseline
- [ ] Note training time + parameter count vs the baseline

## 4. Code learning (packages & methods)

- **`tensorflow.keras`** — `Sequential`, `layers.Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`, `model.compile/fit/evaluate`
- **`numpy`** — input reshaping to `(N,28,28,1)`

➡️ **After we implement:** you explain why the CNN beats logistic regression and what pooling buys us. I'll explain how Keras implements `Conv2D` (the cross-correlation, often lowered to a matrix multiply via im2col / cuDNN) and how backprop flows through it.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — baseline comparison, seeded, OOP-wrapped where it transfers to the main model.

> 🔀 **Note — Ablation-Driven Architecture:** This mini-CNN previews the config-driven `ModelBuilder` (#34); its layers/hyperparameters belong in `config.yaml`. See `CONTRIBUTING.md` §3.
