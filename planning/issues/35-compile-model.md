---
title: "[Phase 4] Compile the model (loss / optimizer / metrics from config)"
labels: ["phase-4-model", "model"]
---
## 1. Description

Compile the assembled model: select the optimizer, loss, and metrics from `config.yaml` via dispatch. This is where the training objective and the gradient-update rule are wired in — small file, big consequences.

## 2. Learning Objective

- **The loss function:** why `categorical_crossentropy` is the natural loss for one-hot multi-class classification.
- **Optimizers compared:** SGD (+momentum) vs Adam vs RMSprop — how each adapts the step size.
- **Learning rate:** the single most important hyperparameter, and why it's config-driven.
- **Metrics vs loss:** the loss is what's optimized; accuracy/macro-F1 are what we *report*.

## 3. To-Do list for coding

- [ ] `models/compile.py` → `build_optimizer(cfg)` dispatch on `model.optimizer` + `model.learning_rate`
- [ ] `compile_model(model, cfg)` setting loss + metrics (accuracy; categorical)
- [ ] `tests/test_compile.py`: returns the right optimizer type with the configured LR; unknown optimizer raises

## 4. Code learning (packages & methods)

- **`tensorflow.keras.optimizers`** — `Adam`, `SGD`, `RMSprop`
- **`tensorflow.keras.losses`** — `CategoricalCrossentropy`
- **`model.compile`** — wiring optimizer/loss/metrics

➡️ **After we implement:** you explain why categorical-crossentropy fits softmax outputs and what the learning rate controls. I'll explain how Adam maintains per-parameter first/second moment estimates to adapt each weight's step.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — dispatch + tests, type hints, raise on unknown option.

> 🔀 **Note — Ablation-Driven Architecture:** Optimizer, LR, loss, metrics are all `config.yaml` values with options commented — prime tuning/ablation knobs. See `CONTRIBUTING.md` §3.
