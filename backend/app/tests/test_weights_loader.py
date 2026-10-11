"""Unit tests for ``app.services.weights_loader.resolve_weights``.

These tests never hit the network — the downloader is injected as a stub.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.core.config import BackendSettings
from app.services.weights_loader import (
    PINNED_ARTIFACT_SHA256,
    PINNED_HUB_REVISIONS,
    ArtifactIntegrityError,
    pinned_sha256,
    resolve_hub_commit,
    resolve_weights,
    sha256_file,
)

PRODUCTION_REPO = "apoorvrajdev/captioning-inceptionv3-transformer"
COMMIT = "0123456789abcdef0123456789abcdef01234567"
UNPINNED_COMMIT = "fedcba9876543210fedcba9876543210fedcba98"
V1_COMMIT = "d917a47decc1f352e50ba4226e92ca5f00d47e9a"
V2_COMMIT = "59d93b4babb16b0ac81eef598f3abc271a355cbf"

# Distinct bytes per file, so a swapped or truncated file changes its digest.
ARTIFACT_BYTES = {
    "model.h5": b"\x89HDF\r\n\x1a\nfake weights",
    "captioning.weights.h5": b"\x89HDF\r\n\x1a\nother fake weights",
    "vocab.json": json.dumps(["", "[UNK]", "a", "dog"]).encode(),
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture(autouse=True)
def _pin_test_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin ``COMMIT`` of the test repo to the digests of ``ARTIFACT_BYTES``."""
    monkeypatch.setitem(
        PINNED_ARTIFACT_SHA256,
        ("user/captioning-weights", COMMIT),
        {name: _sha256(data) for name, data in ARTIFACT_BYTES.items()},
    )


def _snapshot(directory: Path, *names: str) -> Path:
    """Create a fake snapshot directory holding the given files' test bytes."""
    directory.mkdir(parents=True)
    for name in names:
        (directory / name).write_bytes(ARTIFACT_BYTES[name])
    return directory


def _flip_last_byte(path: Path) -> None:
    data = bytearray(path.read_bytes())
    data[-1] ^= 0x01
    path.write_bytes(bytes(data))


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
    """The Space's ``v2.0.0`` variable keeps working, but the tag never reaches the Hub.

    The stub serves test bytes, so the real ``v2.0.0`` digests reject them
    after the download.
    """
    fake_snapshot = _snapshot(tmp_path / "snap", "model.h5", "vocab.json")
    calls: list[dict[str, object]] = []

    settings = BackendSettings(weights_hub_repo=PRODUCTION_REPO, weights_hub_revision="v2.0.0")

    with pytest.raises(ArtifactIntegrityError):
        resolve_weights(settings, downloader=_recording_downloader(fake_snapshot, calls))

    assert [call["revision"] for call in calls] == [V2_COMMIT]
    assert [call["allow_patterns"] for call in calls] == [["model.h5", "vocab.json"]]


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


# ---- SHA-256 verification (TASK-029) ---------------------------------------


def test_sha256_file_matches_hashlib_across_chunk_boundaries(tmp_path: Path) -> None:
    data = bytes(range(256)) * 41  # 10,496 bytes: not a multiple of the chunk size
    path = tmp_path / "blob"
    path.write_bytes(data)
    (tmp_path / "empty").write_bytes(b"")

    assert sha256_file(path, chunk_size=1000) == hashlib.sha256(data).hexdigest()
    assert sha256_file(path) == hashlib.sha256(data).hexdigest()
    assert sha256_file(tmp_path / "empty") == hashlib.sha256(b"").hexdigest()


@pytest.mark.parametrize("tampered", ["model.h5", "vocab.json"])
def test_a_one_byte_change_in_either_file_is_rejected_by_name(
    tmp_path: Path, tampered: str
) -> None:
    fake_snapshot = _snapshot(tmp_path / "snap", "model.h5", "vocab.json")
    _flip_last_byte(fake_snapshot / tampered)
    intact = "vocab.json" if tampered == "model.h5" else "model.h5"
    settings = BackendSettings(
        weights_hub_repo="user/captioning-weights", weights_hub_revision=COMMIT
    )

    with pytest.raises(ArtifactIntegrityError) as excinfo:
        resolve_weights(settings, downloader=_recording_downloader(fake_snapshot, []))

    message = str(excinfo.value)
    actual = _sha256((fake_snapshot / tampered).read_bytes())
    expected = _sha256(ARTIFACT_BYTES[tampered])
    assert f"{tampered} has SHA-256 {actual}, expected {expected}" in message
    assert f"Hub commit {COMMIT} of user/captioning-weights failed verification" in message
    assert "Nothing was loaded" in message
    assert f"{intact} has SHA-256" not in message


def test_every_mismatching_file_is_reported_not_just_the_first(tmp_path: Path) -> None:
    fake_snapshot = _snapshot(tmp_path / "snap", "model.h5", "vocab.json")
    _flip_last_byte(fake_snapshot / "model.h5")
    _flip_last_byte(fake_snapshot / "vocab.json")
    settings = BackendSettings(
        weights_hub_repo="user/captioning-weights", weights_hub_revision=COMMIT
    )

    with pytest.raises(ArtifactIntegrityError, match="model.h5 has SHA-256 .*vocab.json has"):
        resolve_weights(settings, downloader=_recording_downloader(fake_snapshot, []))


def test_a_file_replaced_by_other_pinned_bytes_is_rejected(tmp_path: Path) -> None:
    """Each file is checked against its own digest, not against any pinned digest."""
    fake_snapshot = _snapshot(tmp_path / "snap", "model.h5", "vocab.json")
    (fake_snapshot / "model.h5").write_bytes(ARTIFACT_BYTES["captioning.weights.h5"])
    settings = BackendSettings(
        weights_hub_repo="user/captioning-weights", weights_hub_revision=COMMIT
    )

    with pytest.raises(ArtifactIntegrityError, match="model.h5 has SHA-256"):
        resolve_weights(settings, downloader=_recording_downloader(fake_snapshot, []))


@pytest.mark.parametrize(
    ("repo_id", "revision"),
    [
        ("user/captioning-weights", UNPINNED_COMMIT),  # a valid commit id with no digests
        (PRODUCTION_REPO, UNPINNED_COMMIT),
        (PRODUCTION_REPO, COMMIT),  # pinned only for the test repo
        ("someone-else/captioning-weights", V2_COMMIT),  # pinned only for the production repo
    ],
)
def test_commits_without_pinned_digests_are_refused_before_any_download(
    tmp_path: Path, repo_id: str, revision: str
) -> None:
    calls: list[dict[str, object]] = []
    settings = BackendSettings(weights_hub_repo=repo_id, weights_hub_revision=revision)

    with pytest.raises(ValueError, match=f"Hub commit {revision} of {repo_id} has no SHA-256"):
        resolve_weights(settings, downloader=_recording_downloader(tmp_path, calls))

    assert calls == []


def test_a_pinned_tag_whose_commit_lost_its_digests_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delitem(PINNED_ARTIFACT_SHA256, (PRODUCTION_REPO, V2_COMMIT))
    calls: list[dict[str, object]] = []
    settings = BackendSettings(weights_hub_repo=PRODUCTION_REPO, weights_hub_revision="v2.0.0")

    with pytest.raises(ValueError, match=f"Hub commit {V2_COMMIT} .* has no SHA-256 digests"):
        resolve_weights(settings, downloader=_recording_downloader(tmp_path, calls))

    assert calls == []


@pytest.mark.parametrize(
    ("entry", "match"),
    [
        ({"model.h5": _sha256(b"x")}, "no SHA-256 for vocab.json"),
        ({"vocab.json": _sha256(b"x")}, "no SHA-256 for model.h5"),
        ({}, "no SHA-256 for model.h5"),
        ({"model.h5": _sha256(b"x").upper(), "vocab.json": _sha256(b"x")}, "malformed"),
        ({"model.h5": _sha256(b"x")[:-1], "vocab.json": _sha256(b"x")}, "malformed"),
        ({"model.h5": "g" * 64, "vocab.json": _sha256(b"x")}, "malformed"),
        ({"model.h5": _sha256(b"x"), "vocab.json": ""}, "malformed"),
    ],
)
def test_incomplete_or_malformed_digest_entries_are_refused_before_any_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: dict[str, str], match: str
) -> None:
    monkeypatch.setitem(PINNED_ARTIFACT_SHA256, ("user/captioning-weights", COMMIT), entry)
    calls: list[dict[str, object]] = []
    settings = BackendSettings(
        weights_hub_repo="user/captioning-weights", weights_hub_revision=COMMIT
    )

    with pytest.raises(ValueError, match=match):
        resolve_weights(settings, downloader=_recording_downloader(tmp_path, calls))

    assert calls == []


def test_a_weights_filename_without_a_digest_is_refused_before_any_download(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []
    settings = BackendSettings(
        weights_hub_repo="user/captioning-weights",
        weights_hub_revision=COMMIT,
        weights_hub_filename="model.keras",
    )

    with pytest.raises(ValueError, match="no SHA-256 for model.keras"):
        resolve_weights(settings, downloader=_recording_downloader(tmp_path, calls))

    assert calls == []


def test_every_pinned_tag_has_well_formed_digests_for_both_served_files() -> None:
    """The production tag and the rollback tag both stay servable."""
    assert {tag for _, tag in PINNED_HUB_REVISIONS} >= {"v1.0.0", "v2.0.0"}
    digests: list[str] = []
    for (repo_id, _tag), commit in PINNED_HUB_REVISIONS.items():
        expected = pinned_sha256(repo_id, commit, ["model.h5", "vocab.json"])
        assert sorted(expected) == ["model.h5", "vocab.json"]
        digests.extend(expected.values())
    assert len(set(digests)) == len(digests)  # no revision can pass for another


def test_rollback_vocabulary_digest_matches_the_copy_committed_in_this_repo() -> None:
    """Ties one pinned digest to bytes committed here long before the pin.

    Git stores this file with LF line endings. A Windows checkout with
    ``core.autocrlf`` writes CRLF, so that conversion is undone first.
    """
    committed = Path(__file__).resolve().parents[3] / "models" / "v1.0.0" / "vocab.json"
    data = committed.read_bytes().replace(b"\r\n", b"\n")

    assert _sha256(data) == PINNED_ARTIFACT_SHA256[(PRODUCTION_REPO, V1_COMMIT)]["vocab.json"]


@pytest.mark.parametrize(("tag", "commit"), [("v1.0.0", V1_COMMIT), ("v2.0.0", V2_COMMIT)])
def test_each_pinned_tag_is_checked_against_its_own_commit_digests(
    tmp_path: Path, tag: str, commit: str
) -> None:
    fake_snapshot = _snapshot(tmp_path / "snap", "model.h5", "vocab.json")
    settings = BackendSettings(weights_hub_repo=PRODUCTION_REPO, weights_hub_revision=tag)

    with pytest.raises(ArtifactIntegrityError) as excinfo:
        resolve_weights(settings, downloader=_recording_downloader(fake_snapshot, []))

    message = str(excinfo.value)
    own = PINNED_ARTIFACT_SHA256[(PRODUCTION_REPO, commit)]
    assert f"Hub commit {commit} of {PRODUCTION_REPO}" in message
    assert f"expected {own['model.h5']}" in message
    assert f"expected {own['vocab.json']}" in message
