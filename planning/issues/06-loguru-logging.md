---
title: "[Phase 0] Loguru logging setup"
labels: ["phase-0-foundations", "setup"]
---
## 1. Description

Configure a single, project-wide **Loguru** logger (console + rotating file sink) that every module imports. This enforces the *"logging over printing"* rule and gives us observable pipeline runs and training metrics — essential when comparing ablation experiments.

## 2. Learning Objective

- **Logging vs printing:** why `print` is unstructured and unfilterable, while a logger carries level, timestamp, module, and can fan out to multiple destinations.
- **Log levels:** `DEBUG / INFO / WARNING / ERROR / CRITICAL` and when each is appropriate.
- **Observability in pipelines:** how good logs let you reconstruct *what ran with what config* after the fact.
- **The one exception:** `predict.py` / `predict_live_stream.py` must still `print` their exact required stdout for the audit.

## 3. To-Do list for coding

- [ ] `utils/logging.py` → `setup_logging(cfg: dict) -> None`: remove default sink, add a `stderr` sink and a rotating file sink under `logs/`, format with time+level+module
- [ ] A convenience re-export so modules do `from emotion_detector.utils.logging import logger`
- [ ] Call `setup_logging` at the top of each script entrypoint
- [ ] `tests/test_logging.py`: `setup_logging` runs without error and writes a log line

## 4. Code learning (packages & methods)

- **`loguru`** — `logger`, `logger.remove`, `logger.add(sink, level, rotation, format)`, `logger.info/warning/error`
- **`sys`** — `sys.stderr` as a sink
- **`pathlib`** — ensure `logs/` exists

➡️ **After we implement:** you explain which log level fits which situation and why we keep a file sink. I'll explain how Loguru's `add` registers sinks and how its `rotation` argument rolls log files by size/time.

---

> 📋 **Note — Contributing principles:** Follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) §5 — Loguru everywhere, never `print` (except the two required scripts' stdout).

> 🔀 **Note — Ablation-Driven Architecture:** Log the active config (which stages are on, which strategies chosen) at the start of every run so each ablation experiment is self-documenting. See `CONTRIBUTING.md` §3.
