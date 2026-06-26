---
title: "[Phase 2] Cleaning impl-a — duplicates & corrupt/constant images (config-driven)"
labels: ["phase-2-cleaning", "data"]
---
## 1. Description

Implement the first cleaning strategies behind the Ablation-Driven dispatch: remove exact duplicates and drop corrupt/constant (all-same-pixel) or malformed images — all controlled by `config.yaml` and skippable when the `data_cleaning` stage is off.

## 2. Learning Objective

- **From analysis to action:** translating the §2 findings into deterministic, testable transforms.
- **Idempotent, order-independent cleaning:** a cleaning step applied twice should equal applying it once.
- **Why constant/blank images hurt:** they carry no signal and can destabilize batch statistics.
- **Pipeline-as-composition:** each cleaner is a small object the orchestrator chains.

## 3. To-Do list for coding

- [ ] `data/cleaning.py` → `DuplicateRemover`, `CorruptImageRemover` (each a small strategy class)
- [ ] `build_cleaners(cfg) -> list` dispatch reading `cleaning.duplicates` and stage toggle
- [ ] `is_stage_on(cfg, "data_cleaning")` gate → no-op pass-through when off
- [ ] Log how many rows each step removed
- [ ] `tests/test_cleaning.py`: duplicates removed; constant image dropped; stage-off returns data unchanged

## 4. Code learning (packages & methods)

- **`pandas`** — `drop_duplicates`, boolean masking, `reset_index`
- **`numpy`** — `np.ptp` / `std` to detect constant images, `tobytes` for hashing
- **`hashlib`** — exact-duplicate fingerprints

➡️ **After we implement:** you explain why the stage toggle makes this safely skippable and how the dispatch picks cleaners. I'll explain how `drop_duplicates` hashes rows internally to find repeats in near-linear time.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — strategy classes, type hints, try/except (`KeyError`, `ValueError`), tests incl. the stage-off path.

> 🔀 **Note — Ablation-Driven Architecture:** Reads `cleaning.duplicates` + the `data_cleaning` toggle; turning the stage off must leave the data untouched. See `CONTRIBUTING.md` §3.
