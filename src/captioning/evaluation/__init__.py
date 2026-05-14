"""Evaluation — caption-quality metrics + per-sample diagnostics.

Available metrics (all corpus-level, 0-100 scale where applicable):
    * BLEU-1..4 — :mod:`bleu`
    * ROUGE-L   — :mod:`rouge`
    * METEOR    — :mod:`meteor`  (requires a JRE on PATH)
    * CIDEr     — :mod:`cider`   (requires >= 2 examples)

:func:`compute_all_metrics` in :mod:`runner` is the single entry point used
by the CLI and by future Phase 3 benchmark comparisons; per-sample
diagnostics live in :mod:`inspection`.
"""

from captioning.evaluation.benchmark import RunMeta, write_run_artifacts
from captioning.evaluation.bleu import (
    BleuBreakdown,
    corpus_bleu_breakdown,
    corpus_bleu_score,
)
from captioning.evaluation.cider import MIN_SAMPLES_FOR_CIDER, corpus_cider_score
from captioning.evaluation.inspection import (
    SampleDiagnostics,
    diagnose_many,
    diagnose_sample,
    format_diagnostic_row,
    write_diagnostics_jsonl,
)
from captioning.evaluation.meteor import corpus_meteor_score
from captioning.evaluation.rouge import corpus_rouge_l_score
from captioning.evaluation.runner import MetricsReport, compute_all_metrics

__all__ = [
    "MIN_SAMPLES_FOR_CIDER",
    "BleuBreakdown",
    "MetricsReport",
    "RunMeta",
    "SampleDiagnostics",
    "compute_all_metrics",
    "corpus_bleu_breakdown",
    "corpus_bleu_score",
    "corpus_cider_score",
    "corpus_meteor_score",
    "corpus_rouge_l_score",
    "diagnose_many",
    "diagnose_sample",
    "format_diagnostic_row",
    "write_diagnostics_jsonl",
    "write_run_artifacts",
]
