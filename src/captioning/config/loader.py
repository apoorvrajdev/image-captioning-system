"""YAML-to-Pydantic config loader.

Why this exists separately from ``schema.py``:
    * Schema is *what* a valid config looks like; loader is *how* you build one.
      Splitting them lets tests build an ``AppConfig`` programmatically without
      touching disk, and lets the loader gain features (env-file resolution,
      multi-file merging) without changing the schema.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from captioning.config.schema import AppConfig


def load_config(path: str | Path) -> AppConfig:
    """Load a YAML file into an ``AppConfig`` and validate it.

    Args:
        path: Path to a YAML file with the structure::

            data: {...}
            model: {...}
            train: {...}
            serve: {...}

    Returns:
        A fully validated, immutable ``AppConfig`` instance.

    Raises:
        FileNotFoundError: If the YAML path does not exist.
        pydantic.ValidationError: If any field fails validation.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open(encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    return AppConfig(**raw)
