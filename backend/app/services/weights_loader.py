"""Resolve weights and tokenizer paths, optionally pulling from HuggingFace Hub.

In production we don't want to bake a 158 MB ``.h5`` file into the Docker
image — it makes builds slow and couples weight rotation to image rebuilds.
Instead, the image carries only code, and the runtime pulls the weights file
and ``vocab.json`` from a public Hub repo, at an immutable commit, the first
time the container boots. On HuggingFace Spaces the cache persists across
restarts.

When ``BackendSettings.weights_hub_repo`` is unset we fall back to the
local paths declared in settings, which is what unit tests and `make serve`
use today.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

from app.core.config import BackendSettings
from captioning.preprocessing.tokenizer import VOCAB_JSON_FILENAME
from captioning.utils import get_logger

log = get_logger(__name__)

# A full Hub commit id. Branches and tags can be moved to another commit; a
# commit id can't.
_COMMIT_ID = re.compile(r"[0-9a-f]{40}")

# Published tags and the commit each one named when it was evaluated and
# deployed (checked against the Hub API on 2026-10-12). The loader downloads the
# commit recorded here, never the tag, so moving a tag on the Hub changes
# nothing. A new tag needs a new line here before a Space variable can name it.
PINNED_HUB_REVISIONS: dict[tuple[str, str], str] = {
    (
        "apoorvrajdev/captioning-inceptionv3-transformer",
        "v1.0.0",
    ): "d917a47decc1f352e50ba4226e92ca5f00d47e9a",
    (
        "apoorvrajdev/captioning-inceptionv3-transformer",
        "v2.0.0",
    ): "59d93b4babb16b0ac81eef598f3abc271a355cbf",
}


class SnapshotDownloader(Protocol):
    """Minimal callable shape we need from ``huggingface_hub.snapshot_download``."""

    def __call__(
        self,
        *,
        repo_id: str,
        revision: str,
        cache_dir: str | None,
        allow_patterns: list[str],
    ) -> str: ...


def resolve_hub_commit(repo_id: str, revision: str) -> str:
    """Return the immutable commit id to download for ``revision``.

    A full 40-character commit id is used as given. A tag listed in
    ``PINNED_HUB_REVISIONS`` for ``repo_id`` resolves to its recorded commit.
    Anything else, including a branch such as ``main``, raises ``ValueError``.
    """
    if _COMMIT_ID.fullmatch(revision):
        return revision
    pinned = PINNED_HUB_REVISIONS.get((repo_id, revision))
    if pinned is None:
        raise ValueError(
            f"BACKEND_WEIGHTS_HUB_REVISION={revision!r} is neither a full 40-character "
            f"commit id nor a tag of {repo_id} pinned in "
            "app.services.weights_loader.PINNED_HUB_REVISIONS. Set it to the commit id "
            "(HfApi().model_info(repo_id, revision=tag).sha) or pin the tag first."
        )
    return pinned


def resolve_weights(
    settings: BackendSettings,
    downloader: SnapshotDownloader | None = None,
) -> tuple[Path, Path]:
    """Return ``(weights_path, tokenizer_dir)`` for the predictor to load.

    Local mode (``weights_hub_repo`` is None): returns the paths verbatim.

    Hub mode: resolves the configured revision to a commit id, downloads only
    the weights file and ``vocab.json`` at that commit, then returns paths
    inside the snapshot directory. The downloader is injectable so tests can
    substitute a stub instead of hitting the network.

    Raises:
        ValueError: The revision is not a commit id or a pinned tag.
        FileNotFoundError: The commit lacks the weights file or ``vocab.json``.
    """
    if not settings.weights_hub_repo:
        log.info(
            "weights_source_local",
            weights=str(settings.weights_path),
            tokenizer_dir=str(settings.tokenizer_dir),
        )
        return settings.weights_path, settings.tokenizer_dir

    commit = resolve_hub_commit(settings.weights_hub_repo, settings.weights_hub_revision)
    required = [settings.weights_hub_filename, VOCAB_JSON_FILENAME]

    if downloader is None:
        from huggingface_hub import snapshot_download as _snapshot_download

        downloader = _snapshot_download
        assert downloader is not None  # for type-checker

    cache_dir = str(settings.weights_cache_dir) if settings.weights_cache_dir else None
    log.info(
        "weights_source_hub",
        repo=settings.weights_hub_repo,
        revision=settings.weights_hub_revision,
        commit=commit,
        files=required,
        cache_dir=cache_dir,
    )
    snapshot_dir = Path(
        downloader(
            repo_id=settings.weights_hub_repo,
            revision=commit,
            cache_dir=cache_dir,
            allow_patterns=required,
        )
    )
    missing = [name for name in required if not (snapshot_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Hub commit {commit} of {settings.weights_hub_repo} has no "
            f"{', '.join(missing)}. Serving needs the weights file and "
            f"{VOCAB_JSON_FILENAME} at the same commit."
        )
    weights_path = snapshot_dir / settings.weights_hub_filename
    log.info(
        "weights_downloaded",
        snapshot_dir=str(snapshot_dir),
        weights=str(weights_path),
    )
    return weights_path, snapshot_dir
