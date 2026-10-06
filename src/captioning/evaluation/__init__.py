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
from captioning.evaluation.comparison import (
    ComparisonError,
    RunRecord,
    build_summary,
    load_run,
    render_markdown,
)
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
from captioning.evaluation.slice import EvalSlice, load_eval_slice, slice_fingerprint

__all__ = [
    "MIN_SAMPLES_FOR_CIDER",
    "BleuBreakdown",
    "ComparisonError",
    "EvalSlice",
    "MetricsReport",
    "RunMeta",
    "RunRecord",
    "SampleDiagnostics",
    "build_summary",
    "compute_all_metrics",
    "corpus_bleu_breakdown",
    "corpus_bleu_score",
    "corpus_cider_score",
    "corpus_meteor_score",
    "corpus_rouge_l_score",
    "diagnose_many",
    "diagnose_sample",
    "format_diagnostic_row",
    "load_eval_slice",
    "load_run",
    "render_markdown",
    "slice_fingerprint",
    "write_diagnostics_jsonl",
    "write_run_artifacts",
]
