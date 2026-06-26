---
title: "[Phase 0] Poetry + virtual-environment setup"
labels: ["phase-0-foundations", "setup"]
---
## 1. Description

Initialize the project with **Poetry** inside a dedicated **virtual environment** so the toolchain is isolated and reproducible. This is the foundation for the audit *"Does the environment contain all libraries and their versions?"* — here we just stand up the environment; dependencies come in #03.

## 2. Learning Objective

- **Virtual environments:** what isolation means, how a venv shadows the system Python via `PATH`, and why we never install project deps into the global interpreter.
- **What Poetry is:** a dependency manager + build tool that uses a single `pyproject.toml` and a `poetry.lock` for deterministic installs.
- **Semantic version constraints:** the meaning of `^1.2.3` (compatible-with) vs `~1.2.3` (patch-only) vs pinned `==`.
- **Lockfile determinism:** why `poetry.lock` guarantees every machine resolves the *exact* same versions.

## 3. To-Do list for coding

- [ ] Install Poetry; run `poetry init` (interactive) to create `pyproject.toml`
- [ ] Pin the Python version (e.g. `python = "^3.11"`)
- [ ] `poetry config virtualenvs.in-project true` so the venv lives in `.venv/`
- [ ] Add a `[tool.black]` table with `line-length = 88`
- [ ] `poetry install` to materialize the empty environment
- [ ] Verify with `poetry env info` and `poetry run python --version`

## 4. Code learning (packages & methods)

- **Poetry CLI** — `init`, `config`, `install`, `env info`, `run`
- **`pyproject.toml`** — `[tool.poetry]`, `[tool.poetry.dependencies]`, `[tool.black]`

➡️ **After we implement:** you explain the difference between `^` and `~` constraints and why the lockfile matters. I'll explain how Poetry's resolver treats version selection as a constraint-satisfaction problem.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — Poetry-managed env, Black `line-length = 88`, reproducibility.

> 🔀 **Note — Ablation-Driven Architecture:** A clean, reproducible environment is what lets ablation runs (toggling stages) be compared fairly — same deps, same versions. See `CONTRIBUTING.md` §3.
