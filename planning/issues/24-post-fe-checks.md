---
title: "[Phase 3] Post-feature-engineering checks (before/after distributions, augmented samples)"
labels: ["phase-3-features", "notebook"]
---
## 1. Description

Return to the notebook and verify the feature engineering did what we intended: compare pixel-intensity distributions before vs after normalization/equalization, and render augmented sample images to confirm they're realistic and label-preserving.

## 2. Learning Objective

- **Verifying transforms visually and statistically:** a normalized set should have the expected range; an equalized set a flatter histogram.
- **Reading augmented samples:** confirming flips/rotations stay within "still a natural face".
- **Catching mistakes early:** e.g. double-normalization, wrong dtype, over-aggressive augmentation.

## 3. To-Do list for coding

- [ ] Notebook cells: histograms before vs after normalization and equalization
- [ ] Grid of original vs augmented versions of the same faces
- [ ] Confirm value ranges/dtypes match expectations (`[0,1]` float or `uint8`)
- [ ] Log/annotate anything off for a quick fix

## 4. Code learning (packages & methods)

- **`matplotlib`** — `hist`, `imshow`, `subplots`
- **`numpy`** — range/dtype assertions
- **the augmenter from #22** — to render samples

➡️ **After we implement:** you explain how the histograms confirm normalization worked. I'll explain why a flatter post-equalization histogram is exactly the expected signature of the CDF remap.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — reproducible (seeded) sample selection, clear figures.

> 🔀 **Note — Ablation-Driven Architecture:** Run these checks with FE on vs off to *see* the difference the stage makes before trusting it in training. See `CONTRIBUTING.md` §3.
