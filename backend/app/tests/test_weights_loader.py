"""Unit tests for ``app.services.weights_loader.resolve_weights``.

These tests never hit the network — the downloader is injected as a stub.
"""

from __future__ import annotations

from pathlib import Path

from app.core.config import BackendSettings
from app.services.weights_loader import resolve_weights


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
    fake_snapshot = tmp_path / "snapshots" / "abc123"
    fake_snapshot.mkdir(parents=True)
    calls: list[dict[str, object]] = []

    def fake_downloader(*, repo_id: str, revision: str, cache_dir: str | None) -> str:
        calls.append({"repo_id": repo_id, "revision": revision, "cache_dir": cache_dir})
        return str(fake_snapshot)

    settings = BackendSettings(
        weights_hub_repo="user/captioning-weights",
        weights_hub_revision="v1.0.0",
        weights_hub_filename="model.h5",
        weights_cache_dir=tmp_path / "cache",
    )

    weights_path, tokenizer_dir = resolve_weights(settings, downloader=fake_downloader)

    assert calls == [
        {
            "repo_id": "user/captioning-weights",
            "revision": "v1.0.0",
            "cache_dir": str(tmp_path / "cache"),
        }
    ]
    assert weights_path == fake_snapshot / "model.h5"
    assert tokenizer_dir == fake_snapshot


def test_resolve_weights_hub_mode_passes_none_cache_dir_when_unset(tmp_path: Path) -> None:
    fake_snapshot = tmp_path / "snap"
    fake_snapshot.mkdir()
    seen_cache_dir: list[str | None] = []

    def fake_downloader(*, repo_id: str, revision: str, cache_dir: str | None) -> str:
        seen_cache_dir.append(cache_dir)
        return str(fake_snapshot)

    settings = BackendSettings(
        weights_hub_repo="user/captioning-weights",
        weights_cache_dir=None,
    )

    resolve_weights(settings, downloader=fake_downloader)

    assert seen_cache_dir == [None]


def test_resolve_weights_hub_mode_honors_custom_weights_filename(tmp_path: Path) -> None:
    fake_snapshot = tmp_path / "snap"
    fake_snapshot.mkdir()

    def fake_downloader(*, repo_id: str, revision: str, cache_dir: str | None) -> str:
        return str(fake_snapshot)

    settings = BackendSettings(
        weights_hub_repo="user/captioning-weights",
        weights_hub_filename="captioning.weights.h5",
    )

    weights_path, _ = resolve_weights(settings, downloader=fake_downloader)

    assert weights_path == fake_snapshot / "captioning.weights.h5"
