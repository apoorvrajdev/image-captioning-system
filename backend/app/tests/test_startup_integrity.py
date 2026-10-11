"""Startup never hands an unverified file to the model or tokenizer loader (TASK-029).

These tests run the real ``lifespan`` and the real ``resolve_weights`` with its
default downloader. Only ``huggingface_hub.snapshot_download`` (no network) and
``CaptionPredictor`` (no TensorFlow) are replaced. The stand-in predictor reads
the files the way the real one does: the vocabulary through ``read_vocabulary``,
which ``CaptionTokenizer.load`` calls first, and the weights file's bytes.

Importing ``app.main`` stays TensorFlow-free only while ``captioning.inference``
imports TensorFlow inside functions. If that changes, move this module to
``tests/unit`` so the backend suite stays TensorFlow-free.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, ClassVar

import anyio
import huggingface_hub
import pytest
from fastapi import FastAPI

import app.main as app_main
from app.core.config import BackendSettings
from app.services.weights_loader import PINNED_ARTIFACT_SHA256, ArtifactIntegrityError
from captioning.config.schema import AppConfig
from captioning.preprocessing.tokenizer import read_vocabulary

REPO = "user/captioning-weights"
COMMIT = "0123456789abcdef0123456789abcdef01234567"
UNPINNED_COMMIT = "fedcba9876543210fedcba9876543210fedcba98"
WEIGHTS = b"\x89HDF\r\n\x1a\nfake weights"
VOCAB = ["", "[UNK]", "a", "dog", "on", "grass"]


class RecordingPredictor:
    """Stands in for ``CaptionPredictor``; records every file it consumes."""

    consumed: ClassVar[list[str]] = []
    vocabulary: ClassVar[list[str]] = []

    @classmethod
    def from_artifacts(
        cls, *, weights_path: Path, tokenizer_dir: Path, config: AppConfig
    ) -> RecordingPredictor:
        cls.vocabulary = read_vocabulary(tokenizer_dir, config.model.vocabulary_size)
        cls.consumed.append("vocab.json")
        Path(weights_path).read_bytes()
        cls.consumed.append(Path(weights_path).name)
        return cls()

    def warmup(self) -> None:
        pass


@pytest.fixture
def snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A snapshot holding the pinned bytes, served by a no-network downloader."""
    directory = tmp_path / "snapshots" / COMMIT
    directory.mkdir(parents=True)
    vocab_bytes = json.dumps(VOCAB).encode()
    (directory / "model.h5").write_bytes(WEIGHTS)
    (directory / "vocab.json").write_bytes(vocab_bytes)
    monkeypatch.setitem(
        PINNED_ARTIFACT_SHA256,
        (REPO, COMMIT),
        {
            "model.h5": hashlib.sha256(WEIGHTS).hexdigest(),
            "vocab.json": hashlib.sha256(vocab_bytes).hexdigest(),
        },
    )
    return directory


@pytest.fixture
def downloads(snapshot: Path, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_snapshot_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(app_main, "CaptionPredictor", RecordingPredictor)
    monkeypatch.setattr(RecordingPredictor, "consumed", [])
    monkeypatch.setattr(RecordingPredictor, "vocabulary", [])
    return calls


def _start(revision: str = COMMIT) -> FastAPI:
    """Run the lifespan's startup and shutdown; return the app as it was while serving."""
    app = FastAPI()
    app.state.backend_settings = BackendSettings(
        weights_hub_repo=REPO, weights_hub_revision=revision, warmup=True
    )
    app.state.app_config = AppConfig()
    app.state.predictor_service = None
    seen: dict[str, Any] = {}

    async def run() -> None:
        async with app_main.lifespan(app):
            seen["service"] = app.state.predictor_service

    anyio.run(run)
    app.state.seen_service = seen["service"]
    return app


def test_verified_files_load(downloads: list[dict[str, Any]]) -> None:
    app = _start()

    assert app.state.seen_service is not None
    assert RecordingPredictor.vocabulary == VOCAB
    assert RecordingPredictor.consumed == ["vocab.json", "model.h5"]
    assert [call["allow_patterns"] for call in downloads] == [["model.h5", "vocab.json"]]
    assert [call["revision"] for call in downloads] == [COMMIT]


def test_a_tampered_weights_file_stops_startup_before_the_model_loads(
    snapshot: Path, downloads: list[dict[str, Any]]
) -> None:
    (snapshot / "model.h5").write_bytes(WEIGHTS + b"\x00")

    with pytest.raises(ArtifactIntegrityError, match="model.h5 has SHA-256"):
        _start()

    assert RecordingPredictor.consumed == []


def test_a_tampered_vocabulary_is_never_read_even_when_it_would_validate(
    snapshot: Path, downloads: list[dict[str, Any]]
) -> None:
    """Swapping two tokens keeps the list valid but shifts token ids."""
    swapped = VOCAB[:2] + [VOCAB[3], VOCAB[2]] + VOCAB[4:]
    (snapshot / "vocab.json").write_bytes(json.dumps(swapped).encode())
    assert read_vocabulary(snapshot, AppConfig().model.vocabulary_size) == swapped

    with pytest.raises(ArtifactIntegrityError, match="vocab.json has SHA-256"):
        _start()

    assert RecordingPredictor.consumed == []


def test_an_unpinned_commit_stops_startup_before_any_download(
    downloads: list[dict[str, Any]],
) -> None:
    with pytest.raises(ValueError, match="has no SHA-256 digests"):
        _start(UNPINNED_COMMIT)

    assert downloads == []
    assert RecordingPredictor.consumed == []
