"""Reproducibility helpers.

Why this matters: the IEEE notebook's ``random.shuffle`` of image keys (cell 11)
is non-deterministic without a seed, which means the same code can produce a
different train/val split on every run — and therefore different BLEU. Pinning
the seed makes results reproducible across machines and dates.
"""

from __future__ import annotations

import os
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    pass


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and TensorFlow RNGs from a single integer.

    TF's seeding has multiple layers (``tf.random.set_seed`` for graph-level,
    ``os.environ['PYTHONHASHSEED']`` for hash randomisation, and op-level seeds
    for individual ops). We set as many as practical without forcing TF's
    deterministic mode (which can hurt training throughput by ~15%).

    Args:
        seed: Any non-negative integer.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    # Imported lazily so the utils package doesn't pull NumPy at import time
    # for unrelated callers (e.g. config validation).
    import numpy as np

    np.random.seed(seed)

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
    except ImportError:  # pragma: no cover
        # TF is an optional dep at the *utility* layer; ML callers always have it.
        pass
