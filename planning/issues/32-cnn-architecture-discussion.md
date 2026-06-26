---
title: "[Phase 4] CNN architecture discussion (conv / pool / activation / BN / dropout, VGG-style)"
labels: ["phase-4-model", "documentation"]
---
## 1. Description

A design conversation (and a short design doc) before building: decide the from-scratch CNN architecture for 48×48 grayscale emotion faces. We read the VGG-style reference, discuss each layer type's role, and settle on a concrete block structure, activations, regularization, and output head — justifying every choice.

## 2. Learning Objective

- **Convolutional layers:** learnable filters, receptive fields, parameter sharing, feature maps.
- **Pooling:** spatial downsampling, translation invariance, max vs average.
- **Activations:** why ReLU (non-saturating, sparse) over sigmoid/tanh; softmax at the output.
- **Batch normalization:** stabilizing/standardizing activations to speed and steady training.
- **Dropout:** stochastic regularization to fight overfitting (a core project goal).
- **VGG philosophy:** stacks of small 3×3 convs + doubling channels with depth.

## 3. To-Do list for coding

- [ ] Write `results/model/final_emotion_model_arch.txt` (draft) describing the planned architecture and rationale
- [ ] Decide block pattern: `[Conv→BN→ReLU]×k → MaxPool → Dropout`, channel schedule (e.g. 32→64→128), dense head size, output `Dense(7, softmax)`
- [ ] List the candidate architectures to ablate (`simple_cnn`, `vgg_small`, `resnet_mini`) for `config.yaml`
- [ ] Record the *why* for each decision (to be expanded with iteration history in #45)

## 4. Code learning (packages & methods)

- *Design issue* — references `tensorflow.keras.layers` (`Conv2D`, `BatchNormalization`, `ReLU`, `MaxPooling2D`, `Dropout`, `Dense`) that #33–#34 will assemble.

➡️ **After we implement:** you explain, layer by layer, why this architecture suits 48×48 faces. I'll explain the receptive-field arithmetic (how stacked 3×3 convs grow the effective field) and why BN+ReLU ordering matters.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — document the architecture and the reasoning (the audit asks *why* it was chosen).

> 🔀 **Note — Ablation-Driven Architecture:** Every architectural knob (depth, channels, dropout rate, which architecture) becomes a `config.yaml` option so we can ablate it. See `CONTRIBUTING.md` §3.
