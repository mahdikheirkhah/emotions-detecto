---
title: "[Phase 4] TensorBoard integration + capture tensorboard.png"
labels: ["phase-4-model", "model"]
---
## 1. Description

Wire in **TensorBoard** (mandatory per the subject) so every training run logs scalars (loss/accuracy/LR) and the graph, and capture a `results/model/tensorboard.png` screenshot during training. This satisfies the audit *"Does the screenshot show the usage of TensorBoard to monitor the training?"*

## 2. Learning Objective

- **Why monitor training live:** spotting divergence, overfitting, and dead/exploding gradients as they happen.
- **What TensorBoard shows:** scalar curves (train vs val), the model graph, histograms of weights.
- **Reading train-vs-val curves in real time:** the moment they separate is the overfitting onset.
- **Experiment tracking:** per-run log dirs let you overlay ablation runs and compare.

## 3. To-Do list for coding

- [ ] Add `TensorBoard(log_dir=...)` to `build_callbacks` with a per-run timestamped/config-tagged dir
- [ ] Read `log_dir` root from `config.yaml`; include key config values in the run name
- [ ] Document how to launch: `tensorboard --logdir results/logs`
- [ ] During a real training run, save `results/model/tensorboard.png`

## 4. Code learning (packages & methods)

- **`tensorflow.keras.callbacks`** — `TensorBoard(log_dir, histogram_freq, write_graph)`
- **`tensorboard`** — the `--logdir` viewer
- **`datetime`** — timestamped run directories

➡️ **After we implement:** you explain what to look for in the live curves to catch overfitting. I'll explain how the TensorBoard callback writes event files (protobuf summaries) that the viewer tails and renders.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) §8 — TensorBoard is mandatory; keep the screenshot artifact.

> 🔀 **Note — Ablation-Driven Architecture:** Tag each run's log dir with the active config (stages on/off, architecture) so TensorBoard becomes the side-by-side ablation dashboard. See `CONTRIBUTING.md` §3.
