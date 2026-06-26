---
title: "[Phase 7] Preprocess — detect → crop → 48×48 grayscale → save images (preprocess.py)"
labels: ["phase-7-video", "video"]
---
## 1. Description

Implement the functional preprocessing pipeline graded by `preprocessing_test`: take a **≥20-second** face video, and for each sampled frame **detect → crop → center → convert to 48×48 grayscale**, saving **≥20** images to `results/preprocessing_test/`. This is the bridge that makes webcam frames look exactly like FER-2013 training samples.

## 2. Learning Objective

- **Closing the train/inference gap:** the live image must match the model's training format (48×48 grayscale, face-centered) or accuracy collapses.
- **Crop & center on the detected box:** turning a bounding box into a square, centered face crop.
- **Resize & color conversion:** interpolation when resizing to 48×48; BGR→grayscale.
- **The exact deliverable contract:** 20 (or 21) images from a 20-second clip.

## 3. To-Do list for coding

- [ ] `video/preprocess.py` → `FacePreprocessor.process_frame(frame) -> np.ndarray(48,48)` (detect via `BaseFaceDetector`, crop, center, `cvtColor` gray, `resize`)
- [ ] `scripts/preprocess.py` → sample ~1 frame/sec from the input video, save `image0.png … imageN.png`
- [ ] Apply the **same** normalization/equalization used in training (reuse the fitted preprocessor)
- [ ] `tests/test_preprocess.py`: a ≥20s sample video yields ≥20 saved 48×48 grayscale images, each containing a face

## 4. Code learning (packages & methods)

- **`cv2`** — `cvtColor(BGR2GRAY)`, `resize(interpolation=...)`, `imwrite`
- **`numpy`** — crop slicing, centering math
- **project** — `BaseFaceDetector` + the fitted `BaseImagePreprocessor`

➡️ **After we implement:** you explain why the live crop must match the training format and how centering is computed. I'll explain how `cv2.resize` interpolates pixels (bilinear/area) and why "area" is preferred when downscaling.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) §9 — this is the functional `preprocessing_test`; cover the no-face / bad-frame cases.

> 🔀 **Note — Ablation-Driven Architecture:** The detector, sampling rate, and image transforms are all config-driven and shared with training. See `CONTRIBUTING.md` §3.
