---
title: "[Phase 3] FE-a — normalization strategies (rescale / standardize) (toggleable)"
labels: ["phase-3-features", "data"]
---
## 1. Description

Implement pixel-value normalization as selectable strategies behind the dispatch: `none`, `rescale` (÷255 → [0,1]), and `standardize` (z-score per dataset). This is the first feature-engineering step and is fully toggleable via the `feature_engineering` stage switch.

## 2. Learning Objective

- **Why normalization matters for CNNs:** keeping inputs in a small, centered range stabilizes gradients and speeds convergence.
- **Rescale vs standardize:** `[0,1]` scaling vs zero-mean/unit-variance, and when each is preferred.
- **Train-statistics only:** standardization mean/std must be computed on **train** and reused on val/test (no leakage).
- **For images, "features" = pixels:** unlike tabular data, FE here is mostly intensity transforms.

## 3. To-Do list for coding

- [ ] `data/preprocessing.py` → `IdentityPreprocessor`, `RescalePreprocessor`, `StandardizePreprocessor` (all `BaseImagePreprocessor`)
- [ ] `build_normalizer(cfg) -> BaseImagePreprocessor` dispatch on `preprocessing.normalization`
- [ ] `StandardizePreprocessor.fit(train)` stores mean/std; `transform` applies them
- [ ] Gate on `is_stage_on(cfg, "feature_engineering")`
- [ ] `tests/test_preprocessing.py`: rescale → max ≤ 1.0; standardize → ~0 mean on train; unknown option raises

## 4. Code learning (packages & methods)

- **`numpy`** — `array / 255.0`, `mean`, `std`, broadcasting
- **`abc`** — implements `BaseImagePreprocessor`

➡️ **After we implement:** you explain why mean/std come from train only and what rescale does to gradients. I'll explain how NumPy broadcasting applies a scalar/array op across the whole image tensor without copying.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) §4/§8 — `BaseImagePreprocessor` subclass, fit-on-train-only, tests.

> 🔀 **Note — Ablation-Driven Architecture:** `preprocessing.normalization` lists all options as comments; the whole FE stage is toggleable to get a raw-pixel baseline. See `CONTRIBUTING.md` §3.
