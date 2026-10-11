"""Unit tests for ``app.services.weights_loader.resolve_weights``.

These tests never hit the network — the downloader is injected as a stub.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import BackendSettings
from app.services.weights_loader import (
    PINNED_HUB_REVISIONS,
    resolve_hub_commit,
    resolve_weights,
)

PRODUCTION_REPO = "apoorvrajdev/captioning-inceptionv3-transformer"
COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _snapshot(directory: Path, *names: str) -> Path:
    """Create a fake snapshot directory holding the given (empty) files."""
    directory.mkdir(parents=True)
    for name in names:
        (directory / name).write_bytes(b"")
    return directory


def _recording_downloader(snapshot: Path, calls: list[dict[str, object]]):
    def fake_downloader(
        *, repo_id: str, revision: str, cache_dir: str | None, allow_patterns: list[str]
    ) -> str:
        calls.append(
            {
                "repo_id": repo_id,
                "revision": revision,
                "cache_dir": cache_dir,
                "allow_patterns": allow_patterns,
            }
        )
        return str(snapshot)

    return fake_downloader


def test_resolve_weights_local_mode_returns_settings_paths_verbatim() -> None:
    settings = BackendSettings(
        weights_path=Path("models/v1.0.0/model.h5"),
        tokenizer_dir=Path("models/v1.0.0"),
        weights_hub_repo=None,
    )

    weights_path, tokenizer_dir = resolve_weights(settings, downloader=None)

    assert weights_path == Path("models/v1.0.0/model.h5")
    assert tokenizer_dir == Path("models/v1.0.0")


def test_resolve_weights_hub_mode_calls_downloader_with_expected_args(tmp_path: Path) -> None:
    fake_snapshot = _snapshot(tmp_path / "snapshots" / COMMIT, "model.h5", "vocab.json")
    calls: list[dict[str, object]] = []

    settings = BackendSettings(
        weights_hub_repo="user/captioning-weights",
        weights_hub_revision=COMMIT,
        weights_hub_filename="model.h5",
        weights_cache_dir=tmp_path / "cache",
    )

    weights_path, tokenizer_dir = resolve_weights(
        settings, downloader=_recording_downloader(fake_snapshot, calls)
    )

    assert calls == [
        {
            "repo_id": "user/captioning-weights",
            "revision": COMMIT,
            "cache_dir": str(tmp_path / "cache"),
            "allow_patterns": ["model.h5", "vocab.json"],
        }
    ]
    assert weights_path == fake_snapshot / "model.h5"
    assert tokenizer_dir == fake_snapshot


def test_resolve_weights_hub_mode_passes_none_cache_dir_when_unset(tmp_path: Path) -> None:
    fake_snapshot = _snapshot(tmp_path / "snap", "model.h5", "vocab.json")
    calls: list[dict[str, object]] = []

    settings = BackendSettings(
        weights_hub_repo="user/captioning-weights",
        weights_hub_revision=COMMIT,
        weights_cache_dir=None,
    )

    resolve_weights(settings, downloader=_recording_downloader(fake_snapshot, calls))

    assert [call["cache_dir"] for call in calls] == [None]


def test_resolve_weights_hub_mode_honors_custom_weights_filename(tmp_path: Path) -> None:
    fake_snapshot = _snapshot(tmp_path / "snap", "captioning.weights.h5", "vocab.json")
    calls: list[dict[str, object]] = []

    settings = BackendSettings(
        weights_hub_repo="user/captioning-weights",
        weights_hub_revision=COMMIT,
        weights_hub_filename="captioning.weights.h5",
    )

    weights_path, _ = resolve_weights(
        settings, downloader=_recording_downloader(fake_snapshot, calls)
    )

    assert weights_path == fake_snapshot / "captioning.weights.h5"
    assert calls[0]["allow_patterns"] == ["captioning.weights.h5", "vocab.json"]


def test_pinned_tag_downloads_its_recorded_commit_not_the_tag(tmp_path: Path) -> None:
    """The Space's ``v2.0.0`` variable keeps working, but the tag never reaches the Hub."""
    fake_snapshot = _snapshot(tmp_path / "snap", "model.h5", "vocab.json")
    calls: list[dict[str, object]] = []

    settings = BackendSettings(weights_hub_repo=PRODUCTION_REPO, weights_hub_revision="v2.0.0")

    resolve_weights(settings, downloader=_recording_downloader(fake_snapshot, calls))

    assert [call["revision"] for call in calls] == ["59d93b4babb16b0ac81eef598f3abc271a355cbf"]


@pytest.mark.parametrize(
    ("repo_id", "revision"),
    [
        ("user/captioning-weights", "main"),  # the settings default: a branch
        (PRODUCTION_REPO, "main"),
        (PRODUCTION_REPO, "v3.0.0"),  # a tag nobody pinned
        (PRODUCTION_REPO, "59d93b4"),  # a short commit id
        (PRODUCTION_REPO, "59D93B4BABB16B0AC81EEF598F3ABC271A355CBF"),  # not the Hub's form
        ("someone-else/captioning-weights", "v2.0.0"),  # pinned only for the production repo
        (PRODUCTION_REPO, ""),
    ],
)
def test_unpinned_revisions_are_refused_before_any_download(
    tmp_path: Path, repo_id: str, revision: str
) -> None:
    calls: list[dict[str, object]] = []
    settings = BackendSettings(weights_hub_repo=repo_id, weights_hub_revision=revision)

    with pytest.raises(ValueError, match="neither a full 40-character commit id nor a tag"):
        resolve_weights(settings, downloader=_recording_downloader(tmp_path, calls))

    assert calls == []


@pytest.mark.parametrize(
    ("present", "missing"),
    [(["model.h5"], "vocab.json"), (["vocab.json"], "model.h5"), ([], "model.h5, vocab.json")],
)
def test_missing_artifacts_at_the_commit_fail_by_name(
    tmp_path: Path, present: list[str], missing: str
) -> None:
    fake_snapshot = _snapshot(tmp_path / "snap", *present)
    settings = BackendSettings(
        weights_hub_repo="user/captioning-weights", weights_hub_revision=COMMIT
    )

    with pytest.raises(
        FileNotFoundError,
        match=f"Hub commit {COMMIT} of user/captioning-weights has no {missing}\\.",
    ):
        resolve_weights(settings, downloader=_recording_downloader(fake_snapshot, []))


def test_every_pinned_revision_is_itself_a_valid_commit_id() -> None:
    assert PINNED_HUB_REVISIONS
    for (repo_id, tag), commit in PINNED_HUB_REVISIONS.items():
        assert resolve_hub_commit(repo_id, commit) == commit
        assert resolve_hub_commit(repo_id, tag) == commit
