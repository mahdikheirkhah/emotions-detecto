---
title: "[Phase 7] Face detector-b — DNN face detector via cv2 (DnnFaceDetector) + dispatch"
labels: ["phase-7-video", "video"]
---
## 1. Description

Add a second `BaseFaceDetector` using OpenCV's **DNN** face module (an SSD/ResNet caffe model) for more robust detection under pose/lighting variation, and wire the `face_detector` dispatch so `haar | dnn` is a one-line config switch. This is a clean ablation: compare detectors on the same video.

## 2. Learning Objective

- **Classic vs deep detectors:** why a CNN-based detector generalizes better to angles/lighting than Haar.
- **The blob preprocessing step:** mean subtraction + scaling + resizing into the network's input format.
- **Confidence thresholding:** filtering detections by score and what the threshold trades off.
- **`cv2.dnn` as a runtime:** running pre-trained models from other frameworks inside OpenCV.

## 3. To-Do list for coding

- [ ] `video/dnn.py` → `DnnFaceDetector(BaseFaceDetector)` loading the prototxt + caffemodel
- [ ] `detect(frame)` via `blobFromImage` → `net.setInput` → `net.forward`, filter by `confidence`
- [ ] `build_face_detector(cfg)` dispatch on `face_detector` (`haar | dnn`)
- [ ] `tests/test_dnn.py`: detects a face in the sample image; respects the confidence threshold

## 4. Code learning (packages & methods)

- **`cv2.dnn`** — `readNetFromCaffe`, `blobFromImage(scalefactor, size, mean)`, `net.setInput`, `net.forward`
- **`numpy`** — parse the detection tensor, scale boxes back to frame size

➡️ **After we implement:** you compare Haar vs DNN detections on the same clip and explain the difference. I'll explain what `blobFromImage` does (NCHW layout, mean subtraction) and how the SSD outputs boxes + confidences in one forward pass.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — `BaseFaceDetector` subclass, dispatch, tests, `cv2.error` handling.

> 🔀 **Note — Ablation-Driven Architecture:** `face_detector` toggles `haar | dnn` — a direct ablation of detection quality's effect on emotion accuracy. See `CONTRIBUTING.md` §3.
