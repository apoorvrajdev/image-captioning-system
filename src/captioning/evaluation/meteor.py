"""METEOR (Metric for Evaluation of Translation with Explicit Ordering).

METEOR is part of the standard COCO captioning report alongside BLEU, ROUGE-L,
and CIDEr. It complements BLEU by rewarding semantic matches (synonyms,
stems) rather than only surface n-gram overlap.

Implementation notes:
    * We use the ``pycocoevalcap`` METEOR adapter, which shells out to the
      original Java implementation. METEOR therefore needs a JRE on PATH at
      runtime; the import succeeds either way, the Java process is spawned
      lazily on first scoring call.
    * METEOR's process is long-lived and accepts batches over stdin/stdout —
      a single ``compute_score`` call handles the whole corpus in one round
      trip, so this scales to thousands of examples without thrashing the JVM.
"""

from __future__ import annotations

from collections.abc import Sequence

from captioning.evaluation.tokenization import (
    strip_sentinels_many,
    strip_sentinels_references,
)


def corpus_meteor_score(
    predictions: Sequence[str],
    references: Sequence[Sequence[str]],
) -> float:
    """Compute corpus METEOR via ``pycocoevalcap``.

    Args:
        predictions: One generated caption per example.
        references: One *list* of reference captions per example.

    Returns:
        Corpus METEOR in the 0-100 range to match the rest of this package.
        pycocoevalcap returns 0-1; we multiply by 100 for report parity.

    Raises:
        ImportError: If ``pycocoevalcap`` is not installed.
        ValueError: On mismatched lengths.
        RuntimeError: If the Java METEOR process cannot be launched.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions ({len(predictions)}) and references "
            f"({len(references)}) must have the same length"
        )
    if not predictions:
        return 0.0

    try:
        from pycocoevalcap.meteor.meteor import Meteor
    except ImportError as e:
        raise ImportError(
            "pycocoevalcap is required for METEOR evaluation. "
            "Install via `pip install -r requirements-eval.txt`."
        ) from e

    preds = strip_sentinels_many(predictions)
    refs = strip_sentinels_references(references)

    gts = {str(i): [r for r in ref_list if r] for i, ref_list in enumerate(refs)}
    res = {str(i): [p] for i, p in enumerate(preds)}

    scorer = Meteor()
    try:
        score, _ = scorer.compute_score(gts, res)
    except Exception as e:  # — meteor.py raises bare Exceptions
        raise RuntimeError(
            "METEOR scoring failed. METEOR requires a Java runtime on PATH. "
            f"Underlying error: {e}"
        ) from e

    return float(100.0 * score)
