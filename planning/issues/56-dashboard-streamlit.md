---
title: "[Phase 8] Dashboard-b — Streamlit dashboard (live feed, probabilities, history)"
labels: ["phase-8-delivery", "video"]
---
## 1. Description

Build a polished **Streamlit** dashboard fitting the mental-health-app framing: live webcam feed, the current emotion with a probability bar chart across all 7 classes, and a rolling history/timeline of detected emotions. This is the "proper dashboard" deliverable.

## 2. Learning Objective

- **From script to app:** wrapping the inference pipeline in an interactive UI.
- **Streamlit's rerun model:** how widgets/state drive top-to-bottom re-execution and why `session_state` persists data.
- **Communicating probabilities, not just the top label:** a bar chart over 7 emotions is more honest/clinical.
- **Emotion-over-time:** a timeline that surfaces patterns (the product's actual purpose).

## 3. To-Do list for coding

- [ ] `scripts/dashboard.py` (Streamlit) → webcam/video input, live frame display
- [ ] Show current emotion + a bar chart of all 7 probabilities
- [ ] Maintain a rolling history in `st.session_state`; plot an emotion timeline
- [ ] Read model/source/refresh-rate from `config.yaml`; `streamlit run scripts/dashboard.py`

## 4. Code learning (packages & methods)

- **`streamlit`** — `st.image`, `st.bar_chart`, `st.line_chart`, `st.session_state`, `st.camera_input`/frame loop
- **`pandas`** — history dataframe for charts
- **project** — the #52 preprocessor + trained model

➡️ **After we implement:** you explain Streamlit's rerun + `session_state` model and how the history persists across frames. I'll explain how Streamlit diffs the widget tree each run to update only what changed.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — reuse the pipeline, Loguru behind the UI, type hints.

> 🔀 **Note — Ablation-Driven Architecture:** The dashboard consumes whichever model/config an experiment produced; nothing here is hardcoded. See `CONTRIBUTING.md` §3.
