---
title: "[Phase 7] Real-time-b — load model, predict, print 'HH:MM:SSs : Emotion , XX%' (predict_live_stream.py)"
labels: ["phase-7-video", "video"]
---
## 1. Description

Complete the real-time system: `scripts/predict_live_stream.py` consumes the preprocessed frame stream (#53), runs the trained model on each, and **prints** the timestamped prediction in the exact required format — at least one per second — with a recorded-video fallback. This is the project's headline deliverable.

## 2. Learning Objective

- **End-to-end real-time inference:** capture → detect → preprocess → predict → report, on a clock.
- **Softmax → label + confidence:** turning the 7-way probability vector into "Happy , 73%".
- **Meeting the output contract:** the exact stdout the audit checks, line by line.
- **Latency awareness:** keeping the per-frame pipeline under one second.

## 3. To-Do list for coding

- [ ] `scripts/predict_live_stream.py::main()` — `print("Reading video stream ...")`, load model + config
- [ ] For each `(timestamp, tensor)` from #53: predict, take `argmax` + `max` prob
- [ ] `print(f"{hh:mm:ss}s : {EMOTION} , {prob:.0%}")` (and `print("Preprocessing ...")` per the subject)
- [ ] Recorded-video fallback inherited from #51; verify it runs without a webcam
- [ ] Confirm output matches the required format

## 4. Code learning (packages & methods)

- **`tensorflow.keras.models`** — `load_model`, `model.predict`
- **`numpy`** — `argmax`, `max` on the probability vector
- **`datetime`/`time`** — format the `HH:MM:SS` stamp
- **project** — the #53 preprocessed-frame generator

➡️ **After we implement:** you walk the full live path and explain how a softmax vector becomes the printed label + percentage. I'll explain how `model.predict` batches a single input through the graph and why warm-up/first-call latency happens.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) §5 — this is the second script allowed to `print` its exact required stdout; everything else logs.

> 🔀 **Note — Ablation-Driven Architecture:** Model path, detector, and prediction rate come from `config.yaml`, so this runs any trained model on live or recorded input unchanged. See `CONTRIBUTING.md` §3.
