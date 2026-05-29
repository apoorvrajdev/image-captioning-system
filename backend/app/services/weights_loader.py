"""Resolve weights and tokenizer paths, optionally pulling from HuggingFace Hub.

In production we don't want to bake a 158 MB ``.h5`` file into the Docker
image — it makes builds slow and couples weight rotation to image rebuilds.
Instead, the image carries only code, and the runtime pulls the snapshot
from a public Hub repo (pinned to a revision) the first time the container
boots. On HuggingFace Spaces the cache persists across restarts.

When ``BackendSettings.weights_hub_repo`` is unset we fall back to the
local paths declared in settings, which is what unit tests and `make serve`
use today.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from app.core.config import BackendSettings
from captioning.utils import get_logger

log = get_logger(__name__)


class SnapshotDownloader(Protocol):
    """Minimal callable shape we need from ``huggingface_hub.snapshot_download``."""

    def __call__(
        self,
        *,
        repo_id: str,
        revision: str,
        cache_dir: str | None,
    ) -> str: ...


def resolve_weights(
    settings: BackendSettings,
    downloader: SnapshotDownloader | None = None,
) -> tuple[Path, Path]:
    """Return ``(weights_path, tokenizer_dir)`` for the predictor to load.

    Local mode (``weights_hub_repo`` is None): returns the paths verbatim.

    Hub mode: calls the downloader, then returns paths inside the snapshot
    directory. The downloader is injectable so tests can substitute a stub
    instead of hitting the network.
    """
    if not settings.weights_hub_repo:
        log.info(
            "weights_source_local",
            weights=str(settings.weights_path),
            tokenizer_dir=str(settings.tokenizer_dir),
        )
        return settings.weights_path, settings.tokenizer_dir

    if downloader is None:
        from huggingface_hub import snapshot_download as _snapshot_download

        downloader = _snapshot_download
        assert downloader is not None  # for type-checker

    cache_dir = str(settings.weights_cache_dir) if settings.weights_cache_dir else None
    log.info(
        "weights_source_hub",
        repo=settings.weights_hub_repo,
        revision=settings.weights_hub_revision,
        cache_dir=cache_dir,
    )
    snapshot_dir = Path(
        downloader(
            repo_id=settings.weights_hub_repo,
            revision=settings.weights_hub_revision,
            cache_dir=cache_dir,
        )
    )
    weights_path = snapshot_dir / settings.weights_hub_filename
    log.info(
        "weights_downloaded",
        snapshot_dir=str(snapshot_dir),
        weights=str(weights_path),
    )
    return weights_path, snapshot_dir
