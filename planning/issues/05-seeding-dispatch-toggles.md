---
title: "[Phase 0] Reproducibility seeding + stage-toggle + switch-case dispatch helpers"
labels: ["phase-0-foundations", "config"]
---
## 1. Description

Add the three small utilities the whole pipeline leans on: a global **seed setter** for reproducibility, an `is_stage_on(cfg, name)` helper so any stage can be switched off, and a generic **`dispatch`** helper that maps a config string to a concrete strategy object (the switch-case mechanism).

## 2. Learning Objective

- **Sources of randomness in ML:** weight initialization, data shuffling, train/test splitting, dropout, augmentation — and why each must be seeded for repeatable results.
- **Global seeding:** how one seed makes `random`, NumPy, and TensorFlow produce identical sequences run-to-run.
- **The Strategy pattern:** turning an `if/elif` ladder over a config value into interchangeable polymorphic objects.
- **Ablation via toggles:** how a stage that no-ops when off lets us isolate its effect on accuracy.

## 3. To-Do list for coding

- [ ] `utils/seeding.py` → `set_global_seed(seed: int) -> None` (sets `random.seed`, `numpy.random.seed`, `tf.random.set_seed`, `os.environ["PYTHONHASHSEED"]`)
- [ ] `utils/stages.py` → `is_stage_on(cfg: dict, stage: str) -> bool`
- [ ] `utils/dispatch.py` → `dispatch(name: str, registry: dict[str, Callable]) -> Any` raising `ValueError` on unknown name
- [ ] `tests/test_dispatch.py`: known name returns the right object; unknown name raises `ValueError`

## 4. Code learning (packages & methods)

- **`random`** — `random.seed`
- **`numpy`** — `numpy.random.seed`
- **`tensorflow`** — `tf.random.set_seed`
- **`os`** — `os.environ["PYTHONHASHSEED"]`
- **dict registries** — mapping option strings → constructors

➡️ **After we implement:** you explain why every stochastic step needs the same seed and how the dispatcher selects a strategy. I'll explain how a PRNG (e.g. NumPy's Mersenne-Twister / PCG) turns a single integer seed into a deterministic stream.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — type hints, docstrings, raise on unknown option, tests for both paths.

> 🔀 **Note — Ablation-Driven Architecture:** `dispatch` and `is_stage_on` are the literal switch-case + toggle machinery of `CONTRIBUTING.md` §3 — every later stage uses them.
