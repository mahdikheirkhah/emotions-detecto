---
title: "[Phase 0] Abstract base classes skeleton (the OOP contracts)"
labels: ["phase-0-foundations", "architecture"]
---
## 1. Description

Define the abstract base classes that every concrete strategy in the project will implement. These contracts are what make polymorphic dispatch (#05) possible: callers depend on the base type, the dispatcher returns a base type, and concrete subtypes can be swapped freely via config.

## 2. Learning Objective

- **Abstraction & interfaces:** declaring *what* a component does (`fetch`, `transform`, `detect`, `build`, `predict`) without committing to *how*.
- **The four OOP pillars** in practice (abstraction, inheritance, encapsulation, polymorphism) — the spine of `CONTRIBUTING.md` §4.
- **Why contracts come first:** designing the interface before any implementation keeps the pipeline composable and testable.
- **How Python enforces abstract methods** at instantiation time.

## 3. To-Do list for coding

- [ ] `data/base.py` → `BaseDatasetFetcher` (`fetch`), `BaseImagePreprocessor` (`transform`)
- [ ] `video/base.py` → `BaseFaceDetector` (`detect`)
- [ ] `models/base.py` → `BaseModelBuilder` (`build`), `BaseEmotionClassifier` (`predict`)
- [ ] Each is an `abc.ABC` with `@abstractmethod`s + full docstrings describing the contract and return types
- [ ] `tests/test_bases.py`: instantiating an abstract base directly raises `TypeError`

## 4. Code learning (packages & methods)

- **`abc`** — `ABC`, `@abstractmethod`
- **`typing`** — type aliases for the contracts (e.g. `NDArray`, `Sequence`)

➡️ **After we implement:** you explain why we define bases before concrete classes and how a future face detector "plugs in". I'll explain how `ABCMeta` blocks instantiation of any class that still has unimplemented `@abstractmethod`s.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) §4 — every capability extends an `abc.ABC` base; callers depend on the base, not the concrete subtype.

> 🔀 **Note — Ablation-Driven Architecture:** These bases are the return types of the §3 dispatchers — each config option maps to one concrete subclass of a base defined here.
