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
from pydantic_settings import EnvSettingsSource

from captioning.config.schema import AppConfig


def load_config(path: str | Path) -> AppConfig:
    """Load a YAML file into an ``AppConfig``, apply env overrides, and validate it.

    Precedence, highest first: ``CAPTIONING__*`` environment variables, the
    YAML file, the schema defaults. An override replaces only the field it
    names (``CAPTIONING__SERVE__BEAM_WIDTH=4`` keeps the rest of ``serve``);
    a list value, such as ``CAPTIONING__SERVE__CORS_ALLOWED_ORIGINS``,
    replaces the YAML list whole.

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
        pydantic_settings.sources.SettingsError: If an env value for a list or
            section isn't valid JSON.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open(encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    # pydantic-settings ranks constructor arguments above the environment, so
    # ``AppConfig(**raw)`` alone would let the YAML win. Merge the env values
    # over it first, parsed by the same source ``AppConfig`` uses.
    env = EnvSettingsSource(AppConfig)()
    return AppConfig(**_merge(raw, env))


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return ``base`` updated by ``override``, recursing into sections both set.

    Any other value, lists included, is replaced whole, as pydantic-settings
    merges its own sources.
    """
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged
