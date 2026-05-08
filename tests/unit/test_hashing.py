"""Tests for ``captioning.utils.hashing.sha256_file``."""

from __future__ import annotations

import hashlib
from pathlib import Path

from captioning.utils.hashing import sha256_file


def test_matches_oneshot_hash(tmp_path: Path) -> None:
    """Streaming SHA-256 must equal the one-shot SHA-256."""
    p = tmp_path / "blob.bin"
    payload = b"hello world\n" * 1000
    p.write_bytes(payload)
    assert sha256_file(p) == hashlib.sha256(payload).hexdigest()


def test_handles_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "empty.bin"
    p.touch()
    assert sha256_file(p) == hashlib.sha256(b"").hexdigest()


def test_handles_large_file(tmp_path: Path) -> None:
    """Larger than the internal 64 KB chunk to exercise the streaming path."""
    p = tmp_path / "large.bin"
    payload = b"x" * (256 * 1024)  # 256 KB
    p.write_bytes(payload)
    assert sha256_file(p) == hashlib.sha256(payload).hexdigest()
