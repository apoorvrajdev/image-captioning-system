"""Corpus ROUGE-L for caption evaluation.

ROUGE-L measures the longest common subsequence between a prediction and its
references and is part of the standard COCO captioning report (BLEU, METEOR,
ROUGE-L, CIDEr).

Implementation notes:
    * We use Google's ``rouge_score`` package (the canonical implementation
      since the original perl scripts were retired). It returns precision /
      recall / fmeasure per (prediction, reference) pair.
    * COCO captions ship up to 5 references per image. We take the maximum
      F-measure across references — same convention as pycocoevalcap.
    * The corpus score is the mean of per-sample F-measures, matching how
      sacrebleu and pycocoevalcap aggregate metrics over a dataset.
"""

from __future__ import annotations

from collections.abc import Sequence

from captioning.evaluation.tokenization import (
    strip_sentinels_many,
    strip_sentinels_references,
)


def corpus_rouge_l_score(
    predictions: Sequence[str],
    references: Sequence[Sequence[str]],
) -> float:
    """Compute corpus ROUGE-L F-measure.

    Args:
        predictions: One generated caption per example.
        references: One *list* of reference captions per example.

    Returns:
        Mean ROUGE-L F-measure across examples, in the 0-100 range to match
        sacrebleu's convention (so the report shows BLEU/ROUGE/METEOR/CIDEr
        on comparable scales).

    Raises:
        ImportError: If ``rouge_score`` is not installed
            (``pip install -r requirements-eval.txt``).
        ValueError: On mismatched lengths or an empty references slot.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions ({len(predictions)}) and references "
            f"({len(references)}) must have the same length"
        )
    if not predictions:
        return 0.0

    try:
        from rouge_score import rouge_scorer
    except ImportError as e:
        raise ImportError(
            "rouge_score is required for ROUGE-L evaluation. "
            "Install via `pip install -r requirements-eval.txt`."
        ) from e

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    preds = strip_sentinels_many(predictions)
    refs = strip_sentinels_references(references)

    total = 0.0
    for hypothesis, ref_list in zip(preds, refs, strict=True):
        valid_refs = [r for r in ref_list if r]
        if not valid_refs or not hypothesis:
            continue
        best = max(scorer.score(r, hypothesis)["rougeL"].fmeasure for r in valid_refs)
        total += best

    return float(100.0 * total / len(preds))
