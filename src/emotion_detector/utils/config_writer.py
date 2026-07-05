"""Write winning values back into ``config.yaml`` **without losing comments**.

Closing the ablation loop (CONTRIBUTING §3): once a hyperparameter search picks a
winner, those values become the file's new *defaults* so the best run is
reproducible by simply re-running. A plain ``yaml.safe_load`` → ``yaml.dump`` would
strip every ``# options`` menu and the whole ``tuning.search_space`` record — the
experiment would be lost. So this does a **surgical, line-scoped text edit**:
rewrite only the value token of ``block.<key>: <value>`` lines, keeping each line's
inline comment (and its column) intact.

Edits are scoped to a single top-level block, so promoting ``model.*`` winners never
touches the identically-named arrays under ``tuning.search_space`` — that block stays
as the permanent record of what was searched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _format_value(value: Any) -> str:
    """Render *value* as YAML matching the file's style (bools lower, strings quoted)."""
    if isinstance(value, bool):  # bool is an int subclass — check it first
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    return repr(value)  # 64 -> '64', 0.001 -> '0.001'


def _block_span(lines: List[str], block: str) -> Tuple[int, int]:
    """Return ``(start, end)`` line indices spanning a top-level ``block:``.

    ``start`` is the ``block:`` line; ``end`` is the first following top-level line
    (column-0, non-comment) — i.e. the next block — or ``len(lines)``.

    Raises:
        KeyError: if *block* is not a top-level key in the file.
    """
    start = None
    for i, line in enumerate(lines):
        if line.rstrip() == f"{block}:" or line.startswith(f"{block}:"):
            if not line[0].isspace():  # top-level only
                start = i
                break
    if start is None:
        raise KeyError(f"Top-level block '{block}:' not found.")

    end = len(lines)
    for j in range(start + 1, len(lines)):
        line = lines[j]
        if line and not line[0].isspace() and not line.lstrip().startswith("#"):
            end = j
            break
    return start, end


def _find_key_line(lines: List[str], start: int, end: int, key: str) -> Optional[int]:
    """Index of the ``<indent><key>:`` line within ``(start, end)``, or ``None``."""
    for i in range(start + 1, end):
        stripped = lines[i].lstrip()
        if stripped.startswith(f"{key}:") and lines[i] != stripped:  # indented child
            return i
    return None


def _rewrite_value(line: str, key: str, value: Any) -> str:
    """Replace the value on *line*, preserving indent and any inline comment column."""
    indent = line[: len(line) - len(line.lstrip())]
    head = f"{indent}{key}: {_format_value(value)}"
    if "#" in line:  # values never contain '#', so this is the comment
        col = line.index("#")
        pad = max(1, col - len(head))  # keep the comment at its original column
        return head + " " * pad + line[col:]
    return head


def promote_values(config_path: str, block: str, values: Dict[str, Any]) -> List[str]:
    """Overwrite ``block.<key>`` defaults in *config_path* with *values*, in place.

    Preserves every comment and the untouched blocks byte-for-byte. Only keys inside
    the named block are edited, so promoting ``model.*`` winners leaves the
    ``tuning.search_space`` arrays (which share key names) alone.

    Args:
        config_path: Path to the YAML file to edit.
        block:       Top-level block whose children to update (e.g. ``"model"``).
        values:      ``{key: new_value}`` — every key must already exist in *block*.

    Returns:
        The list of keys that were updated (in *values* order).

    Raises:
        KeyError: if the block is missing, or a key is absent from it.
    """
    path = Path(config_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    start, end = _block_span(lines, block)

    updated: List[str] = []
    for key, value in values.items():
        idx = _find_key_line(lines, start, end, key)
        if idx is None:
            raise KeyError(f"Key '{key}' not found under block '{block}:'.")
        lines[idx] = _rewrite_value(lines[idx], key, value)
        updated.append(key)

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return updated
