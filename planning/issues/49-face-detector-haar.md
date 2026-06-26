---
title: "[Phase 7] Face detector-a — Haar cascade via cv2 (HaarFaceDetector)"
labels: ["phase-7-video", "video"]
---
## 1. Description

Implement the first `BaseFaceDetector` using OpenCV's **Haar cascade** classifier (imported via `cv2`, as the audit requires). Given a frame, it returns face bounding boxes. This is the workhorse that lets us crop faces from webcam frames.

## 2. Learning Objective

- **How face detection works:** scanning the image at many positions/scales and classifying each window as face / not-face.
- **Haar features + integral image:** simple light/dark rectangle features computed in O(1) via an integral image.
- **The cascade idea:** a chain of increasingly strict classifiers that rejects non-faces early (fast).
- **Detector parameters:** `scaleFactor`, `minNeighbors`, `minSize` and their precision/recall trade-offs.

## 3. To-Do list for coding

- [ ] `video/haar.py` → `HaarFaceDetector(BaseFaceDetector)` loading `haarcascade_frontalface_default.xml`
- [ ] `detect(frame) -> list[(x, y, w, h)]` via `detectMultiScale`
- [ ] Read `scaleFactor`/`minNeighbors`/`minSize` from `config.yaml`
- [ ] `tests/test_haar.py`: detects a face in a known sample image; returns `[]` on a blank image

## 4. Code learning (packages & methods)

- **`cv2`** — `CascadeClassifier`, `cv2.data.haarcascades`, `detectMultiScale`, `cvtColor` (BGR→gray)
- **`numpy`** — bounding-box handling

➡️ **After we implement:** you explain how `scaleFactor`/`minNeighbors` trade false positives vs misses. I'll explain how the Viola–Jones cascade uses an integral image to evaluate Haar features in constant time per window.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) §4 — `BaseFaceDetector` subclass, `cv2.error`/`FileNotFoundError` handling, tests.

> 🔀 **Note — Ablation-Driven Architecture:** `face_detector: haar` is one dispatch option; cascade params live in `config.yaml`. See `CONTRIBUTING.md` §3.
