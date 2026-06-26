---
title: "[Phase 8] Functional preprocessing_test + CI workflow (GitHub Actions: black + pytest)"
labels: ["phase-8-delivery", "testing"]
---
## 1. Description

Finalize the graded functional test and wire continuous integration: a `preprocessing_test` that proves a ≥20-second face video yields ≥20 correctly-formatted images, plus a GitHub Actions workflow running Black + pytest on every push so the project stays green (per `CONTRIBUTING.md` §1).

## 2. Learning Objective

- **Functional/acceptance testing:** validating the full preprocessing contract end-to-end, not just units.
- **What CI buys you:** automatic formatting + test enforcement on every commit prevents regressions.
- **Reproducible CI environments:** installing from the locked deps so CI matches local.
- **Fast, offline tests in CI:** why we mock the model/GPU and ship a tiny sample video.

## 3. To-Do list for coding

- [ ] `tests/test_preprocessing_test.py` → run `scripts/preprocess.py` on a short sample clip; assert ≥20 images, each 48×48 grayscale and containing a face
- [ ] Commit a small sample input video under `results/preprocessing_test/` (or a fixture)
- [ ] `.github/workflows/ci.yml` → set up Python, install deps, run `black --check .` and `pytest`
- [ ] Confirm the workflow passes on a push

## 4. Code learning (packages & methods)

- **`pytest`** — drive the script, assert on outputs
- **`cv2`** — verify saved image size/channels + re-detect a face
- **GitHub Actions** — `actions/setup-python`, cache, run steps
- **`black`** — `--check` in CI

➡️ **After we implement:** you explain what the functional test guarantees that unit tests don't. I'll explain how a GitHub Actions runner provisions a fresh VM, restores caches, and executes the workflow steps.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) §1/§9 — CI enforces Black + pytest; `preprocessing_test` is the functional gate.

> 🔀 **Note — Ablation-Driven Architecture:** CI runs with a fixed default `config.yaml`; keep tests config-aware so toggles don't silently break the build. See `CONTRIBUTING.md` §3.
