"""Shared pytest fixtures and config.

Keeping fixtures here (rather than per-test) is the standard pytest pattern
and makes `pytest --fixtures` discoverable for new contributors.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from captioning.utils.seed import set_global_seed


@pytest.fixture(autouse=True)
def _seed_everything() -> Iterator[None]:
    """Seed all RNGs before each test for deterministic results."""
    set_global_seed(42)
    yield


@pytest.fixture
def tiny_caption_corpus() -> list[str]:
    """A small, deterministic corpus used by tokenizer tests."""
    return [
        "[start] a man on a surfboard [end]",
        "[start] a dog in the park [end]",
        "[start] two children playing with a ball [end]",
        "[start] a cat sitting on a chair [end]",
        "[start] a man riding a bike on the street [end]",
    ]


@pytest.fixture
def tmp_artifacts_dir(tmp_path: Path) -> Path:
    """A clean temp dir for save/load round-trip tests."""
    return tmp_path / "artifacts"
