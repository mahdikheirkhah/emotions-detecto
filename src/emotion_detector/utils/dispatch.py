"""Switch-case dispatch helper for the Ablation-Driven Architecture."""
from __future__ import annotations

from typing import Any, Callable


def dispatch(name: str, registry: dict[str, Callable[[], Any]]) -> Any:
    """Look up *name* in *registry* and return the result of calling it.

    This is the switch-case mechanism: config strings map to factory
    callables (usually class constructors or factory functions).

    Example::

        registry = {
            "rescale":      RescalePreprocessor,
            "standardize":  StandardizePreprocessor,
            "histogram_eq": HistogramEqualizer,
        }
        preprocessor = dispatch(cfg_get(cfg, "preprocessing.normalization"), registry)

    Args:
        name:     The option string read from config (e.g. "rescale").
        registry: Mapping of option strings to zero-argument callables.

    Returns:
        The object returned by ``registry[name]()``.

    Raises:
        ValueError: if *name* is not a key in *registry*.
    """
    if name not in registry:
        known = ", ".join(sorted(registry))
        raise ValueError(
            f"Unknown option '{name}'. Valid options: {known}"
        )
    return registry[name]()
