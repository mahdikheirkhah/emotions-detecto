---
title: "[Phase 4] Build-a — reusable conv-block builder"
labels: ["phase-4-model", "model"]
---
## 1. Description

Implement a single reusable function/class that builds one convolutional block — `Conv2D → BatchNorm → ReLU → (Conv→BN→ReLU) → MaxPool → Dropout` — parameterized by channels, kernel size, and dropout rate from config. The full network (#34) is just a stack of these.

## 2. Learning Objective

- **Composability:** why expressing the net as repeated blocks keeps it readable and ablatable.
- **The canonical block ordering:** Conv → BN → activation (and why BN goes before the nonlinearity).
- **Parameterization:** channels/kernel/dropout as arguments so the same code yields many architectures.
- **Counting parameters:** how a conv layer's parameter count depends on kernel size × in-channels × out-channels.

## 3. To-Do list for coding

- [ ] `models/blocks.py` → `conv_block(x, filters, kernel_size, dropout, n_convs) -> tensor`
- [ ] Use the Keras functional API so blocks chain cleanly
- [ ] Pull defaults from `config.yaml` (`model.dropout`, `model.kernel_size`)
- [ ] `tests/test_blocks.py`: output spatial dims halve after pool; channel count matches `filters`

## 4. Code learning (packages & methods)

- **`tensorflow.keras.layers`** — `Conv2D`, `BatchNormalization`, `ReLU`/`Activation`, `MaxPooling2D`, `Dropout`
- **Keras functional API** — calling layers on tensors

➡️ **After we implement:** you explain why a block halves spatial size while growing channels. I'll explain how `BatchNormalization` maintains running mean/variance and behaves differently in train vs inference mode.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — small single-purpose builder, type hints, shape tests.

> 🔀 **Note — Ablation-Driven Architecture:** Block hyperparameters (dropout, kernel, #convs) are config values, enabling architecture ablations without editing code. See `CONTRIBUTING.md` §3.
