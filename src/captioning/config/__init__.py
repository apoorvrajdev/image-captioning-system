"""Configuration package — Pydantic schemas and YAML loaders.

Why a dedicated package? Configs are the project's *type system*. Every other
module accepts an `AppConfig` (or a sub-config) instead of pulling globals,
which makes them testable in isolation and trivially overridable in CI / serve.
"""

from captioning.config.loader import load_config
from captioning.config.schema import (
    AppConfig,
    BaselineDecodeConfig,
    CompareConfig,
    ComparedModelConfig,
    DataConfig,
    ModelConfig,
    ServeConfig,
    TrainConfig,
)

__all__ = [
    "AppConfig",
    "BaselineDecodeConfig",
    "CompareConfig",
    "ComparedModelConfig",
    "DataConfig",
    "ModelConfig",
    "ServeConfig",
    "TrainConfig",
    "load_config",
]
