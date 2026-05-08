"""File-hashing helper used by the paper-notebook freeze CI check."""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK = 64 * 1024


def sha256_file(path: str | Path) -> str:
    """Return the hex-digest SHA-256 of a file, streaming 64KB chunks.

    Streaming (rather than ``open(...).read()``) keeps memory bounded for
    notebooks with embedded image outputs that can hit hundreds of MB.
    """
    h = hashlib.sha256()
    path = Path(path)
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()
