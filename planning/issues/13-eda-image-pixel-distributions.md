---
title: "[Phase 1] EDA-c — image grids + pixel-intensity / brightness / contrast distributions"
labels: ["phase-1-data", "notebook"]
---
## 1. Description

Actually *look* at the data: render grids of sample faces per emotion, and analyze pixel-intensity statistics across the dataset (overall histogram, per-image brightness and contrast distributions). This reveals quality issues (too dark/bright, low contrast) that motivate normalization and histogram equalization.

## 2. Learning Objective

- **Visualizing image data:** turning a `48×48` array back into a viewable image and reading it.
- **Pixel-intensity histograms:** what the distribution of brightness tells you about lighting variance.
- **Brightness vs contrast:** brightness ≈ mean intensity, contrast ≈ spread (std) — why both vary wildly in "in-the-wild" faces.
- **Motivating preprocessing:** how these findings justify rescaling and histogram equalization (#20–#21).

## 3. To-Do list for coding

- [ ] In the notebook: plot a grid (e.g. 5×7) of random samples labeled by emotion
- [ ] Global pixel-intensity histogram (all images flattened)
- [ ] Per-image **mean** (brightness) and **std** (contrast) distributions
- [ ] Flag suspiciously dark/bright/low-contrast images for §2/§3

## 4. Code learning (packages & methods)

- **`matplotlib`** — `imshow(cmap="gray")`, `subplots`, `hist`
- **`numpy`** — `mean(axis=...)`, `std(axis=...)`, `flatten`/`ravel`

➡️ **After we implement:** you explain what the brightness/contrast spread implies for a CNN. I'll explain how `imshow` maps array values to grayscale via a colormap + normalization, and what a histogram bin actually counts.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — good visualizations, log findings, seeded sampling for reproducible grids.

> 🔀 **Note — Ablation-Driven Architecture:** Findings here justify the toggleable `preprocessing.normalization` options (`none | rescale | standardize | histogram_eq`). See `CONTRIBUTING.md` §3.
