---
title: "[Phase 8] Dashboard-a — OpenCV overlay window (face box + emotion label)"
labels: ["phase-8-delivery", "video"]
---
## 1. Description

Build the live visual dashboard the project asks for, in OpenCV: a window showing the webcam feed with a rectangle drawn around the detected face and the predicted emotion + confidence overlaid in real time. This is the visible payoff of the whole pipeline.

## 2. Learning Objective

- **Drawing on frames:** annotating a live image with rectangles and text.
- **The render loop:** read → predict → draw → show → handle quit-key, at interactive rates.
- **UX of confidence:** showing probability so a clinician sees how sure the model is.
- **Reusing the pipeline:** the dashboard is just #53/#54 plus drawing — no new ML.

## 3. To-Do list for coding

- [ ] `scripts/predict_live_stream.py` (or a `--display` flag) → draw `cv2.rectangle` on the face + `cv2.putText` label/confidence
- [ ] Main loop with `cv2.imshow` + `cv2.waitKey` quit handling
- [ ] Color-code or sort the label by confidence; config flag to enable display
- [ ] Verify it runs on the recorded-video fallback too

## 4. Code learning (packages & methods)

- **`cv2`** — `rectangle`, `putText`, `imshow`, `waitKey`, `destroyAllWindows`
- **project** — the #53 stream + #54 predictor

➡️ **After we implement:** you explain the read→predict→draw→show loop and the quit handling. I'll explain how `cv2.imshow`/`waitKey` drive the GUI event loop and why `waitKey` is required for the window to refresh.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — reuse existing components, clean resource release.

> 🔀 **Note — Ablation-Driven Architecture:** Display is a config flag layered on the same pipeline; it changes nothing about the model path or preprocessing. See `CONTRIBUTING.md` §3.
