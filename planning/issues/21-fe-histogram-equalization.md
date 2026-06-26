---
title: "[Phase 3] FE-b — histogram equalization / CLAHE (toggleable)"
labels: ["phase-3-features", "data"]
---
## 1. Description

Add contrast enhancement as a selectable preprocessing strategy: global **histogram equalization** and **CLAHE** (Contrast-Limited Adaptive Histogram Equalization). The §2 brightness/contrast findings motivate this — it helps faces shot in poor lighting become more uniform for the CNN.

## 2. Learning Objective

- **What histogram equalization does:** remaps intensities so the cumulative distribution becomes ~uniform, spreading out contrast.
- **Global vs adaptive (CLAHE):** global can over-amplify noise; CLAHE equalizes in tiles with a clip limit to avoid that.
- **When it helps vs hurts:** great for low-contrast faces, can wash out already-balanced ones — hence a toggle.
- **Consistency train↔inference:** the same transform must run in the live webcam pipeline (#52).

## 3. To-Do list for coding

- [ ] `data/preprocessing.py` → `HistogramEqualizer`, `ClaheEqualizer` (`BaseImagePreprocessor`)
- [ ] Extend `build_normalizer`/preprocessing dispatch to include `histogram_eq` and `clahe`
- [ ] Ensure these run on `uint8` grayscale before rescale (order documented)
- [ ] `tests/test_equalization.py`: output stays `48×48` `uint8`; histogram is flatter than input

## 4. Code learning (packages & methods)

- **`cv2`** — `cv2.equalizeHist`, `cv2.createCLAHE(clipLimit, tileGridSize).apply`
- **`numpy`** — histogram comparison for the test

➡️ **After we implement:** you explain when CLAHE beats global equalization. I'll explain the algorithm behind `equalizeHist`: building the intensity histogram, computing its CDF, and using the normalized CDF as the pixel remap function.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — `cv2.error` handling, type hints, tests; identical transform reused at inference.

> 🔀 **Note — Ablation-Driven Architecture:** Adds options to `preprocessing.normalization` (`histogram_eq | clahe`) — never delete prior options. See `CONTRIBUTING.md` §3.
