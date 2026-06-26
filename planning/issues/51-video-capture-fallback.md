---
title: "[Phase 7] Video capture — webcam stream + recorded-video fallback"
labels: ["phase-7-video", "video"]
---
## 1. Description

Build the video-source abstraction: open the webcam with OpenCV, and **fall back to a recorded video file** if the webcam is unavailable (the audit explicitly requires this). Yields frames to the rest of the pipeline.

## 2. Learning Objective

- **How a webcam stream works:** a `VideoCapture` device delivering frames in a read loop.
- **Graceful degradation:** detecting a failed camera and switching to a file without crashing.
- **Frames, FPS, and timing:** the relationship between capture rate and our "≥1 prediction/second" requirement.
- **Resource hygiene:** always releasing the capture device.

## 3. To-Do list for coding

- [ ] `video/capture.py` → `VideoSource` opening `cv2.VideoCapture(index)`; if `isOpened()` is False, open `video.fallback_path`
- [ ] `frames()` generator yielding frames; expose FPS; clean `release()` (context manager)
- [ ] Source/index/fallback path from `config.yaml`
- [ ] `tests/test_capture.py`: with a tiny sample video, frames are yielded; bad index falls back

## 4. Code learning (packages & methods)

- **`cv2`** — `VideoCapture`, `isOpened`, `read`, `get(CAP_PROP_FPS)`, `release`
- **`contextlib`** — context-manager cleanup

➡️ **After we implement:** you explain the fallback logic and why releasing the device matters. I'll explain how `VideoCapture.read` pulls/decodes a frame from the camera/codec and returns it as a BGR NumPy array.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) §7 — try/except around capture, fallback path, always `release`.

> 🔀 **Note — Ablation-Driven Architecture:** Webcam index and fallback video path are config values, so the same pipeline runs on live or recorded input. See `CONTRIBUTING.md` §3.
