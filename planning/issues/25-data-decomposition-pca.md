---
title: "[Phase 3] Data decomposition / PCA (toggleable) + data.md §6"
labels: ["phase-3-features", "data"]
---
## 1. Description

Add an **optional, toggleable** dimensionality-reduction stage (PCA) and document the reasoning in **Section 6** of `data.md`. For a CNN we usually keep raw 48×48 images (convolutions exploit spatial structure PCA would destroy), so this stage defaults **off** — but it's valuable for the MNIST logistic-regression baseline and as a teaching ablation.

## 2. Learning Objective

- **What PCA does:** finds orthogonal directions of maximum variance and projects data onto the top components, reducing dimensionality with minimal information loss.
- **Why PCA + CNN is usually wrong:** flattening to components discards the 2-D spatial layout convolutions rely on.
- **Where PCA *is* useful here:** compressing flattened pixels for a linear/logistic baseline; visualizing class separability in 2-D.
- **Explained variance:** choosing the number of components from the cumulative-variance curve.

## 3. To-Do list for coding

- [ ] `data/decomposition.py` → `PcaReducer` with `fit(train)` / `transform`
- [ ] `build_decomposer(cfg)` dispatch on `stages.decomposition` + `decomposition.n_components`
- [ ] Default the stage **off**; when off, pass data through unchanged
- [ ] Plot cumulative explained variance; (optional) 2-D PCA scatter colored by emotion
- [ ] Add `data.md` Section **`## 6. Data decomposition`** explaining the on/off rationale

## 4. Code learning (packages & methods)

- **`sklearn.decomposition`** — `PCA(n_components).fit/transform`, `explained_variance_ratio_`
- **`numpy`** — flatten images to vectors for PCA
- **`matplotlib`** — explained-variance and scatter plots

➡️ **After we implement:** you explain why we keep PCA off for the CNN but on for the baseline. I'll explain how PCA computes components via the covariance matrix's eigenvectors (or the SVD of the centered data).

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — fit-on-train-only, type hints, tests, documented rationale.

> 🔀 **Note — Ablation-Driven Architecture:** `stages.decomposition` is a clean ablation switch (default `false`) — flipping it shows exactly why spatial structure matters to a CNN. See `CONTRIBUTING.md` §3.
