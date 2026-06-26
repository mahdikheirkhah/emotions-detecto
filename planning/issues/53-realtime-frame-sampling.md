---
title: "[Phase 7] Real-time-a — per-second frame sampling + preprocessing glue"
labels: ["phase-7-video", "video"]
---
## 1. Description

Build the real-time loop's front half: read the video source (#51), sample **at least one frame per second**, and run each through the face-preprocessing pipeline (#52) to produce model-ready 48×48 tensors — the stream of inputs the predictor (#54) consumes.

## 2. Learning Objective

- **Frame-rate vs prediction-rate:** why we don't classify every frame — sampling ~1/sec meets the requirement cheaply.
- **Throughput thinking:** detection + preprocessing per frame must fit inside the time budget.
- **Handling "no face" frames:** skip or hold last prediction, without crashing the loop.
- **Decoupling capture from prediction:** clean hand-off via a generator of preprocessed tensors.

## 3. To-Do list for coding

- [ ] `video/stream.py` → `preprocessed_frames(source, cfg)` generator: sample by elapsed time/FPS, preprocess, yield `(timestamp, tensor)` (or skip if no face)
- [ ] Read `video.predictions_per_second` from config
- [ ] Log "Preprocessing ..." per emitted frame (matching the subject's prompt)
- [ ] `tests/test_stream.py`: a 3s sample video emits ≥3 preprocessed tensors

## 4. Code learning (packages & methods)

- **`cv2`** — `get(CAP_PROP_FPS)`, frame counting for time-based sampling
- **`time`** — wall-clock pacing
- **project** — `VideoSource` + `FacePreprocessor`

➡️ **After we implement:** you explain how we pick ~1 frame/sec and what we do when no face is found. I'll explain how frame timestamps are derived from FPS/frame-index and the trade-offs of time- vs count-based sampling.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — Loguru "Preprocessing ..." logs, robust to missing faces.

> 🔀 **Note — Ablation-Driven Architecture:** `predictions_per_second` and the detector are config values; the same loop serves live or recorded input. See `CONTRIBUTING.md` §3.
