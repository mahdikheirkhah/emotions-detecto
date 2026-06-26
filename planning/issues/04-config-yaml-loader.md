---
title: "[Phase 0] config.yaml schema + loader (the Ablation-Driven backbone)"
labels: ["phase-0-foundations", "config"]
---
## 1. Description

Build the heart of the Ablation-Driven Architecture: a single `config.yaml` that holds **every** decision, strategy, hyperparameter, and constant — each value annotated with **all its options as inline comments** — plus a small typed loader with a dotted-key accessor and fail-loud validation. Every later issue reads from this file instead of hardcoding.

## 2. Learning Objective

- **Configuration-as-code:** separating the *what* (decisions, in config) from the *how* (logic, in code), and why this enables systematic experimentation.
- **What an ablation study is:** changing exactly one factor at a time to measure its contribution — only possible when each factor is a named, switchable config value.
- **Fail-loud design:** why an unknown/missing key should raise immediately rather than silently default.
- **YAML basics:** mappings, lists, scalars, comments — and why YAML (not JSON) so options can be commented inline.

## 3. To-Do list for coding

- [ ] Write `config.yaml` skeleton with sections: `global` (seed, paths), `stages` (toggles), `cleaning`, `preprocessing`, `model`, `tuning`, `evaluation`, `face_detector`, `video` — **each value with a `# options:` comment**
- [ ] `src/emotion_detector/utils/config.py`:
  - [ ] `load_config(path: str) -> dict`
  - [ ] `cfg_get(cfg: dict, dotted_key: str) -> Any` (e.g. `cfg_get(cfg, "model.optimizer")`) that raises `KeyError` on a missing key
- [ ] `tests/test_config.py`: loads the file; missing key raises `KeyError`

## 4. Code learning (packages & methods)

- **`pyyaml`** — `yaml.safe_load`
- **`pathlib`** — `Path.read_text`
- **`functools.reduce` / `operator.getitem`** — walking the dotted path into nested dicts

➡️ **After we implement:** you explain the config sections and why each value carries its option comment. I'll explain how `yaml.safe_load` parses YAML into Python objects and why `safe_load` is used instead of `load` (arbitrary-object construction risk).

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — type hints, docstrings, try/except (`FileNotFoundError`, `KeyError`), a matching test.

> 🔀 **Note — Ablation-Driven Architecture:** This **is** the backbone (`CONTRIBUTING.md` §3). Every option must appear here with a comment listing all alternatives; nothing downstream may hardcode a decision.
