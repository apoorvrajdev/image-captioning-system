"""Resolve weights and tokenizer paths, optionally pulling from HuggingFace Hub.

In production we don't want to bake a 158 MB ``.h5`` file into the Docker
image — it makes builds slow and couples weight rotation to image rebuilds.
Instead, the image carries only code, and the runtime pulls the weights file
and ``vocab.json`` from a public Hub repo, at an immutable commit, the first
time the container boots. On HuggingFace Spaces the cache persists across
restarts. Both files are checked against SHA-256 digests pinned for that
commit before their paths are returned. The loaders reopen them by path, so
the check covers the cache as it was at that moment.

When ``BackendSettings.weights_hub_repo`` is unset we fall back to the
local paths declared in settings, which is what unit tests and `make serve`
use today. Local files aren't hashed.
"""

from __future__ import annotations

import hashlib
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

# A lower-case hex SHA-256 digest, the form hashlib and Git LFS pointers use.
_SHA256 = re.compile(r"[0-9a-f]{64}")

# SHA-256 of every file serving reads, per repo and commit. A commit without an
# entry here, or a file missing from its entry, is refused before any download.
# Provenance (Hub API, read-only, 2026-10-12):
# - model.h5 is a Git LFS file. Its digest is the ``oid sha256`` of the LFS
#   pointer stored in the commit's tree, and that pointer's git blob id matches
#   the tree entry. v2.0.0's digest also appears in docs/EVAL_METHODOLOGY.md
#   (2026-10-06).
# - vocab.json is a plain git file, for which the Hub publishes only a SHA-1
#   blob id. Its digest is the SHA-256 of bytes whose git blob id matched the
#   commit's tree entry (v1.0.0 f8ce580…, v2.0.0 1a437e1…). v1.0.0's also equals
#   the repo's own models/v1.0.0/vocab.json.
# Serving a new commit means adding its digests here, from the same sources,
# and deploying that first.
PINNED_ARTIFACT_SHA256: dict[tuple[str, str], dict[str, str]] = {
    (
        "apoorvrajdev/captioning-inceptionv3-transformer",
        "d917a47decc1f352e50ba4226e92ca5f00d47e9a",  # v1.0.0
    ): {
        "model.h5": "bfe020d920aa2f3d019bf7b5b33904384057372e7c304a9e101a2a59fe110084",
        "vocab.json": "45ec1704d73046303cbd5292590b2e204b194a2d8345dfb84de81370b4ab4eef",
    },
    (
        "apoorvrajdev/captioning-inceptionv3-transformer",
        "59d93b4babb16b0ac81eef598f3abc271a355cbf",  # v2.0.0
    ): {
        "model.h5": "74963a3f7cd01b16f44cd179f1e21e9eb8d60b52d460e517cdcf38c015689b07",
        "vocab.json": "4315c5a7350d6052dbc120e02d218b9a95a23797aa24ec5670824eb6667d2886",
    },
}


class ArtifactIntegrityError(RuntimeError):
    """A downloaded file's SHA-256 differs from the digest pinned for its commit."""


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


def pinned_sha256(repo_id: str, commit: str, filenames: list[str]) -> dict[str, str]:
    """Return the pinned SHA-256 of each of ``filenames`` at ``commit``.

    Raises ``ValueError`` when the commit has no entry in
    ``PINNED_ARTIFACT_SHA256``, when a file has no digest there, or when a
    digest isn't 64 lower-case hex characters. Callers run this before
    downloading, so an unverifiable revision never reaches the network.
    """
    pinned = PINNED_ARTIFACT_SHA256.get((repo_id, commit))
    if pinned is None:
        raise ValueError(
            f"Hub commit {commit} of {repo_id} has no SHA-256 digests in "
            "app.services.weights_loader.PINNED_ARTIFACT_SHA256, so its files can't be "
            "verified. Add the digests for that commit (see the comment above the table) "
            "and deploy them before serving it."
        )
    expected: dict[str, str] = {}
    for name in filenames:
        digest = pinned.get(name)
        if digest is None:
            raise ValueError(
                f"PINNED_ARTIFACT_SHA256 has no SHA-256 for {name} at Hub commit {commit} "
                f"of {repo_id}. Pin a digest for every file serving reads."
            )
        if not _SHA256.fullmatch(digest):
            raise ValueError(
                f"PINNED_ARTIFACT_SHA256 holds a malformed SHA-256 for {name} at Hub commit "
                f"{commit} of {repo_id}: expected 64 lower-case hex characters."
            )
        expected[name] = digest
    return expected


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA-256 of ``path``, read in chunks so memory stays bounded."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(directory: Path, expected: dict[str, str], *, repo_id: str, commit: str) -> None:
    """Check every file in ``expected`` under ``directory`` against its digest.

    Every file is hashed, even after a mismatch, so the error lists each file
    that differs.

    Raises:
        ArtifactIntegrityError: One or more files differ from their digest.
    """
    mismatches = []
    for name, want in expected.items():
        got = sha256_file(directory / name)
        if got != want:
            mismatches.append(f"{name} has SHA-256 {got}, expected {want}")
    if mismatches:
        raise ArtifactIntegrityError(
            f"Hub commit {commit} of {repo_id} failed verification in {directory}: "
            f"{'; '.join(mismatches)}. Nothing was loaded. The download or the local "
            "cache is corrupt or was altered: delete the file from the Hub cache, both "
            "the snapshot entry and the blob it links to, and restart to download it "
            "again. If it still fails, the Hub no longer serves the pinned bytes; don't "
            "serve this revision."
        )


def resolve_weights(
    settings: BackendSettings,
    downloader: SnapshotDownloader | None = None,
) -> tuple[Path, Path]:
    """Return ``(weights_path, tokenizer_dir)`` for the predictor to load.

    Local mode (``weights_hub_repo`` is None): returns the paths verbatim.

    Hub mode: resolves the configured revision to a commit id, looks up the
    SHA-256 pinned for each file at that commit, downloads only the weights
    file and ``vocab.json`` at that commit, checks both against their digests,
    then returns paths inside the snapshot directory. The downloader is
    injectable so tests can substitute a stub instead of hitting the network.

    Raises:
        ValueError: The revision is not a commit id or a pinned tag, or the
            commit has no valid pinned digest for a file.
        FileNotFoundError: The commit lacks the weights file or ``vocab.json``.
        ArtifactIntegrityError: A downloaded file differs from its digest.
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
    expected = pinned_sha256(settings.weights_hub_repo, commit, required)

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
    verify_sha256(snapshot_dir, expected, repo_id=settings.weights_hub_repo, commit=commit)
    weights_path = snapshot_dir / settings.weights_hub_filename
    log.info(
        "weights_verified",
        snapshot_dir=str(snapshot_dir),
        weights=str(weights_path),
        sha256=expected,
    )
    return weights_path, snapshot_dir
