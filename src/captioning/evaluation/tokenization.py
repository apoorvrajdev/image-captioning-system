"""Reference/hypothesis tokenisation helpers shared across metric modules.

CIDEr, METEOR (via pycocoevalcap) and ROUGE-L all expect *string* inputs but
each implements its own tokenisation internally. To compare metrics on the
same footing we strip our sentinel tokens once, up front, before any metric
sees a caption.

These helpers exist because every metric module would otherwise re-implement
the same ``[start]`` / ``[end]`` stripping inline — and bugs would diverge.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from captioning.preprocessing.caption import END_TOKEN, START_TOKEN


def strip_sentinels(caption: str) -> str:
    """Remove ``[start]`` / ``[end]`` sentinels and collapse whitespace.

    Args:
        caption: A caption string that may carry our training sentinels.

    Returns:
        The same caption with the sentinels removed and consecutive whitespace
        collapsed to a single space. Empty input returns ``""``.
    """
    if not caption:
        return ""
    cleaned = caption.replace(START_TOKEN, " ").replace(END_TOKEN, " ")
    return " ".join(cleaned.split())


def strip_sentinels_many(captions: Iterable[str]) -> list[str]:
    """Apply :func:`strip_sentinels` to every caption in ``captions``."""
    return [strip_sentinels(c) for c in captions]


def strip_sentinels_references(
    references: Sequence[Sequence[str]],
) -> list[list[str]]:
    """Apply :func:`strip_sentinels` to every reference list."""
    return [strip_sentinels_many(refs) for refs in references]
