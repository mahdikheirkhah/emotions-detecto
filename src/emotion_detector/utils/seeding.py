"""Global reproducibility seeding for random, NumPy, and TensorFlow."""
from __future__ import annotations

import os
import random


def set_global_seed(seed: int) -> None:
    """Fix all sources of randomness so every run produces identical results.

    Covers: Python's random module, NumPy, TensorFlow, and the hash-seed
    used by Python's built-in hash() for strings/bytes.

    Args:
        seed: Integer seed value (use cfg_get(cfg, "global.seed")).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
