"""CIDEr (Consensus-based Image Description Evaluation) corpus metric.

CIDEr is the metric the COCO captioning leaderboard ranks by. It computes a
TF-IDF weighting over n-grams of the references and measures cosine similarity
to the prediction. Higher is better; correctly trained models score in the
range 0.6 - 1.4 on COCO val.

Implementation notes:
    * We delegate to ``pycocoevalcap`` — the reference implementation used by
      the original CIDEr paper and by every COCO submission.
    * CIDEr's TF-IDF is corpus-level: scoring a *single* example returns 0
      because every n-gram is "common" to that one-document corpus. The
      ``runner`` aggregator and the CLI guard against calling CIDEr with
      fewer than ``MIN_SAMPLES_FOR_CIDER`` examples and return ``None``
      in that case.
"""

from __future__ import annotations

from collections.abc import Sequence

from captioning.evaluation.tokenization import (
    strip_sentinels_many,
    strip_sentinels_references,
)

# CIDEr's TF-IDF is degenerate below this — every n-gram is "common"
# to the entire corpus, so the score collapses to 0. We surface ``None``
# instead of a misleading value below this threshold.
MIN_SAMPLES_FOR_CIDER = 2


def corpus_cider_score(
    predictions: Sequence[str],
    references: Sequence[Sequence[str]],
) -> float:
    """Compute corpus CIDEr.

    Args:
        predictions: One generated caption per example.
        references: One *list* of reference captions per example.

    Returns:
        CIDEr in the 0-10 range (pycocoevalcap convention; the typical COCO
        leaderboard value is in [0, 2]).

    Raises:
        ImportError: If ``pycocoevalcap`` is not installed.
        ValueError: On mismatched lengths or if called with fewer than
            ``MIN_SAMPLES_FOR_CIDER`` examples (in which case CIDEr's TF-IDF
            is degenerate and the score is meaningless).
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions ({len(predictions)}) and references "
            f"({len(references)}) must have the same length"
        )
    if len(predictions) < MIN_SAMPLES_FOR_CIDER:
        raise ValueError(
            f"CIDEr requires at least {MIN_SAMPLES_FOR_CIDER} examples; "
            f"got {len(predictions)}. TF-IDF is degenerate on smaller corpora."
        )

    try:
        from pycocoevalcap.cider.cider import Cider
    except ImportError as e:
        raise ImportError(
            "pycocoevalcap is required for CIDEr evaluation. "
            "Install via `pip install -r requirements-eval.txt`."
        ) from e

    preds = strip_sentinels_many(predictions)
    refs = strip_sentinels_references(references)

    # pycocoevalcap expects {image_id: [captions]} dicts.
    gts = {str(i): [r for r in ref_list if r] for i, ref_list in enumerate(refs)}
    res = {str(i): [p] for i, p in enumerate(preds)}

    scorer = Cider()
    score, _ = scorer.compute_score(gts, res)
    return float(score)
