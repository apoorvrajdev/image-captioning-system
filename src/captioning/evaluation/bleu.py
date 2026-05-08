"""Corpus BLEU score (Phase 1 minimal implementation).

The IEEE paper reports BLEU ~24 on COCO val. The notebook does not include
the evaluation code that produced this number — we add it here so the new
modular pipeline can verify it matches the paper.

Phase 1 ships *one* metric (corpus BLEU-4 via ``sacrebleu``) on purpose:
    * sacrebleu is the de-facto BLEU implementation. NLTK's BLEU has
      idiosyncratic smoothing and produces slightly different numbers; we
      use sacrebleu so the published number is reproducible by anyone with
      pip.
    * Phase 1b expands to BLEU-1..4, CIDEr, METEOR, ROUGE-L, all in this
      package, all behind the same ``runner.py`` interface.
"""

from __future__ import annotations

from collections.abc import Sequence


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

    # sacrebleu's `corpus_bleu` expects parallel lists, one *per reference
    # slot*: refs_by_slot[slot_index][example_index].
    max_refs = max(len(r) for r in references) if references else 0
    refs_by_slot = [
        [refs[i] if i < len(refs) else "" for refs in references] for i in range(max_refs)
    ]

    bleu = sacrebleu.corpus_bleu(list(predictions), refs_by_slot)
    return float(bleu.score)
