"""Evaluation — caption-quality metrics.

Phase 1 ships a corpus-BLEU implementation only; Phase 1b expands to CIDEr,
METEOR, and ROUGE-L (which is why this is its own package, not a single file).
"""

from captioning.evaluation.bleu import corpus_bleu_score

__all__ = ["corpus_bleu_score"]
