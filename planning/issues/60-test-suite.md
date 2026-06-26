---
title: "[Phase 8] Test suite — unit tests (config, data, model utils)"
labels: ["phase-8-delivery", "testing"]
---
## 1. Description

Consolidate and round out the `pytest` suite so the whole pipeline is covered: config loading/validation, dataset parsing, cleaning/preprocessing strategies, splits (no leakage), model builders, and evaluation — including the failure paths (`except` blocks) per the contributing rules.

## 2. Learning Objective

- **Why tests matter in ML code:** silent preprocessing/label bugs are the most expensive kind; tests catch them.
- **Unit vs functional tests:** small isolated checks vs the end-to-end `preprocessing_test` (#61).
- **Mocking heavy I/O:** faking dataset loads and the model so the suite runs fast and offline.
- **Testing the error paths:** asserting the right exception fires on bad input.

## 3. To-Do list for coding

- [ ] Ensure each module from earlier issues has a matching `tests/test_*.py`
- [ ] Cover failure paths: missing config key, malformed pixels, unknown strategy option, empty/constant image, split overlap
- [ ] Use fixtures + monkeypatch/mocks for dataset and model
- [ ] `poetry run pytest` green locally

## 4. Code learning (packages & methods)

- **`pytest`** — `assert`, fixtures, `pytest.raises`, `tmp_path`, parametrize
- **`unittest.mock` / `monkeypatch`** — fake heavy I/O and models
- **`numpy`** — synthetic fixtures

➡️ **After we implement:** you explain which bug each failure-path test guards against. I'll explain how `pytest` collects tests and how `pytest.raises` asserts an exception's type/message.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) §9 — every function has a test; cover main + `except` paths; mock heavy I/O.

> 🔀 **Note — Ablation-Driven Architecture:** Add a test that every config strategy option dispatches to a class and that unknown options raise — guarding the §3 contract. See `CONTRIBUTING.md` §3.
