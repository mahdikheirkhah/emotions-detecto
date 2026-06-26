---
title: "[Phase 6] Model optimization — pruning (optional, toggleable)"
labels: ["phase-6-optional", "optional", "model"]
---
## 1. Description

**Optional:** apply weight **pruning** (zeroing the least-important weights) to produce a smaller, sparser model, then fine-tune to recover accuracy. Toggleable via config; pairs well with quantization (#47).

## 2. Learning Objective

- **What pruning does:** removing low-magnitude weights to create sparsity, reducing model size (and potentially latency with sparse kernels).
- **Magnitude-based pruning:** why small weights are the safest to drop.
- **Prune → fine-tune:** recovering the accuracy lost when connections are cut.
- **Sparsity vs real speedup:** when sparsity actually pays off in practice.

## 3. To-Do list for coding

- [ ] `models/optimize.py` → `prune(model, cfg)` using `tensorflow_model_optimization` polynomial-decay sparsity schedule
- [ ] Config: `optimization.pruning.target_sparsity` + toggle
- [ ] Fine-tune the pruned model; strip pruning wrappers for export
- [ ] Measure size + accuracy before/after; log deltas; test that a pruned model still predicts

## 4. Code learning (packages & methods)

- **`tensorflow_model_optimization`** — `prune_low_magnitude`, `PolynomialDecay`, `strip_pruning`, `UpdatePruningStep` callback
- **`numpy`** — count zero weights / sparsity ratio

➡️ **After we implement:** you explain the prune-then-fine-tune cycle and report the sparsity/accuracy trade-off. I'll explain how the sparsity schedule gradually raises the pruning threshold over training steps.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — optional, measured, documented; verify recovered accuracy.

> 🔀 **Note — Ablation-Driven Architecture:** `optimization.pruning` is a config toggle; compare dense vs pruned models under identical evaluation. See `CONTRIBUTING.md` §3.
