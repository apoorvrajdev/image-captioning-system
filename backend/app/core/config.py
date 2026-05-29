"""Backend runtime settings.

These settings drive the FastAPI process itself: where to find the trained
artifacts, what to advertise as the model version, whether to warm up at
boot. They are intentionally separate from ``captioning.config.AppConfig``,
which owns the *ML* configuration (architecture, decode strategy, CORS
origins). Keeping the two layers split lets ops change deployment paths
without touching research configs, and vice versa.

Override any field via environment variable, prefixed with ``BACKEND_``::

    BACKEND_CONFIG_PATH=configs/base.yaml
    BACKEND_WEIGHTS_PATH=models/v1.0.0/model.h5
    BACKEND_TOKENIZER_DIR=models/v1.0.0
    BACKEND_MODEL_VERSION=v1.0.0
    BACKEND_WARMUP=true
    BACKEND_WEIGHTS_HUB_REPO=your-username/captioning-weights
    BACKEND_WEIGHTS_HUB_REVISION=v1.0.0
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class BackendSettings(BaseSettings):
    """Settings for the FastAPI inference service."""

    config_path: Path = Field(
        default=Path("configs/base.yaml"),
        description="Path to the YAML AppConfig consumed by the ML package.",
    )
    weights_path: Path = Field(
        default=Path("models/v1.0.0/model.h5"),
        description="Path to the trained Keras weights file (used when weights_hub_repo is unset).",
    )
    tokenizer_dir: Path = Field(
        default=Path("models/v1.0.0"),
        description=(
            "Directory containing vocab.pkl / vocab.json artifacts "
            "(used when weights_hub_repo is unset)."
        ),
    )
    model_version: str = Field(
        default="v1.0.0",
        description="Semantic version surfaced in /healthz and caption responses.",
    )
    api_version: str = Field(
        default="0.1.0",
        description="FastAPI app version (shown in OpenAPI docs).",
    )
    warmup: bool = Field(
        default=True,
        description="Run one dummy inference at startup so the first request is fast.",
    )
    request_id_header: str = Field(
        default="x-request-id",
        description="HTTP header used for request correlation IDs.",
    )

    # ---- HuggingFace Hub weights pull (WS-A4) -------------------------------
    # When ``weights_hub_repo`` is set, ``lifespan`` calls
    # ``huggingface_hub.snapshot_download`` and resolves ``weights_path`` and
    # ``tokenizer_dir`` to paths inside the downloaded snapshot. This lets the
    # Docker image stay small and lets us rotate weights without rebuilding.
    weights_hub_repo: str | None = Field(
        default=None,
        description="HuggingFace Hub repo id (e.g. 'user/captioning-weights'). None = use local paths.",
    )
    weights_hub_revision: str = Field(
        default="main",
        description="Git ref/tag/commit to pin (recommended: pin a tag like 'v1.0.0').",
    )
    weights_hub_filename: str = Field(
        default="model.h5",
        description="Filename of the weights file inside the Hub snapshot.",
    )
    weights_cache_dir: Path | None = Field(
        default=None,
        description="Local cache dir for snapshot_download. None = HF Hub default ($HF_HOME).",
    )

    model_config = SettingsConfigDict(
        env_prefix="BACKEND_",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("config_path", "weights_path", "tokenizer_dir")
    @classmethod
    def _expand_user(cls, value: Path) -> Path:
        return value.expanduser()

    @field_validator("weights_cache_dir")
    @classmethod
    def _expand_optional_user(cls, value: Path | None) -> Path | None:
        return value.expanduser() if value is not None else None


@lru_cache(maxsize=1)
def get_backend_settings() -> BackendSettings:
    """Return a process-wide ``BackendSettings`` instance.

    Cached so env-var parsing happens once. Tests that need to override env
    can call ``get_backend_settings.cache_clear()`` between cases.
    """
    return BackendSettings()
