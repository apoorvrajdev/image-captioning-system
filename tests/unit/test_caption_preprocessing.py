"""Tests for ``captioning.preprocessing.caption.preprocess_caption``.

The function is the cheapest possible thing to test thoroughly, and it's also
the hottest train/serve-skew risk: any divergence here changes both the
training vocabulary and the inference path.
"""

from __future__ import annotations

import re

import pytest

from captioning.preprocessing.caption import (
    END_TOKEN,
    START_TOKEN,
    preprocess_caption,
)


def _notebook_baseline(text: str) -> str:
    """Verbatim notebook cell 3 for parity comparison."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return "[start] " + text + " [end]"


@pytest.mark.parametrize(
    "raw",
    [
        "A man riding a bike",
        "ALL CAPS ARE LOWERED",
        "punctuation, removed!",
        "  multiple    spaces ",
        "Numbers 123 stay",
        "Tabs\tand\nnewlines",
        "",
    ],
)
def test_matches_notebook_baseline(raw: str) -> None:
    assert preprocess_caption(raw) == _notebook_baseline(raw)


def test_wraps_in_sentinels() -> None:
    out = preprocess_caption("hello world")
    assert out.startswith(START_TOKEN + " ")
    assert out.endswith(" " + END_TOKEN)


def test_idempotent_on_already_clean() -> None:
    """Already-lowercase, no-punctuation input shouldn't change between
    inner content runs."""
    clean = "a man riding a bike"
    out1 = preprocess_caption(clean)
    # Inner content (without sentinels) should equal the input.
    inner = out1.removeprefix(f"{START_TOKEN} ").removesuffix(f" {END_TOKEN}")
    assert inner == clean


def test_strips_emoji_and_unicode_punct() -> None:
    """``\\w`` in Python regex matches unicode word chars by default; punctuation
    (including emoji) is dropped. Documenting current behaviour."""
    out = preprocess_caption("hello 😀 world!")
    inner = out.removeprefix(f"{START_TOKEN} ").removesuffix(f" {END_TOKEN}")
    # Emoji is non-word non-whitespace → stripped; collapsed spaces leave one space.
    assert inner == "hello world"
