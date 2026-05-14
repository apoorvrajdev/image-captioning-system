"""Corpus BLEU score.

The IEEE paper reports BLEU-4 ~24 on COCO val. ``sacrebleu`` is the de-facto
BLEU implementation; NLTK's BLEU has idiosyncratic smoothing and would not
reproduce the published number across machines.

``corpus_bleu_score`` returns BLEU-4 (the default n=4 score) so existing
callers keep working. ``corpus_bleu_breakdown`` additionally exposes BLEU-1,
BLEU-2, BLEU-3, BLEU-4 in one pass — useful for the inspection utility and
for the JSON report consumed by Phase 3 cross-model comparison.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from captioning.evaluation.tokenization import (
    strip_sentinels_many,
    strip_sentinels_references,
)


@dataclass(frozen=True)
class BleuBreakdown:
    """Per-n BLEU precisions plus the overall BLEU-4 score (0-100 scale)."""

    bleu1: float
    bleu2: float
    bleu3: float
    bleu4: float


def _refs_by_slot(references: Sequence[Sequence[str]]) -> list[list[str]]:
    """Convert ragged per-example references to sacrebleu's per-slot layout."""
    max_refs = max(len(r) for r in references) if references else 0
    return [[refs[i] if i < len(refs) else "" for refs in references] for i in range(max_refs)]


def corpus_bleu_score(
    predictions: Sequence[str],
    references: Sequence[Sequence[str]],
) -> float:
    """Compute corpus BLEU-4 via ``sacrebleu``.

    Args:
        predictions: One generated caption per evaluation example.
        references: One *list* of reference captions per evaluation example.
            COCO has up to 5 references per image; pad shorter lists with the
            empty string ``""`` if needed (sacrebleu handles ragged lists).

    Returns:
        BLEU-4 in the 0-100 range (sacrebleu's convention; multiply by 1
        to compare with NLTK's 0-1 range — they're not interchangeable).

    Raises:
        ImportError: If sacrebleu is not installed. Install via the eval
            extras: ``pip install -e ".[eval]"`` or the requirements file.
    """
    return corpus_bleu_breakdown(predictions, references).bleu4


def corpus_bleu_breakdown(
    predictions: Sequence[str],
    references: Sequence[Sequence[str]],
) -> BleuBreakdown:
    """Compute BLEU-1, BLEU-2, BLEU-3, BLEU-4 in a single pass.

    Args:
        predictions: One generated caption per example.
        references: One *list* of reference captions per example.

    Returns:
        :class:`BleuBreakdown` with all four cumulative BLEU-n scores on the
        0-100 scale (sacrebleu's convention).

    Raises:
        ImportError: If sacrebleu is not installed.
        ValueError: On mismatched lengths.
    """
    try:
        import sacrebleu
    except ImportError as e:
        raise ImportError(
            "sacrebleu is required for BLEU evaluation. "
            "Install it via `pip install -r requirements-eval.txt`."
        ) from e

    if len(predictions) != len(references):
        raise ValueError(
            f"predictions ({len(predictions)}) and references "
            f"({len(references)}) must have the same length"
        )

    preds = strip_sentinels_many(predictions)
    refs = strip_sentinels_references(references)
    refs_by_slot = _refs_by_slot(refs)

    # ``corpus_bleu`` only returns BLEU-4. To get cumulative BLEU-1..3 we
    # instantiate ``BLEU`` directly with ``max_ngram_order=n``, which weights
    # the geometric mean over precisions[:n] (same convention as NLTK's
    # cumulative BLEU and the COCO eval scripts).
    bleu_cls = sacrebleu.metrics.BLEU
    scores: list[float] = []
    for n in (1, 2, 3, 4):
        scorer = bleu_cls(max_ngram_order=n, effective_order=True)
        scores.append(float(scorer.corpus_score(preds, refs_by_slot).score))
    return BleuBreakdown(bleu1=scores[0], bleu2=scores[1], bleu3=scores[2], bleu4=scores[3])
