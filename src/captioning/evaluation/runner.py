"""Single entry point that returns every implemented caption-quality metric.

``compute_all_metrics`` is the shared aggregator used by the CLI
(:mod:`scripts.evaluate`) and the per-sample inspection utility. It produces
a single :class:`MetricsReport` so downstream code never has to know which
metrics exist in the package — only how to read fields off the dataclass.

Adding a new metric is the four-step pattern this package already follows
elsewhere:
    1. Implement ``corpus_<metric>_score`` in a sibling module.
    2. Add an entry to :class:`MetricsReport`.
    3. Call it from :func:`compute_all_metrics` (wrapped in a try/except so a
       single broken metric never poisons the whole report).
    4. Add a unit test on a toy fixture.

The exception swallowing is deliberate — METEOR needs Java, CIDEr needs
multiple samples, sacrebleu is always available. We do NOT want one
unavailable metric to kill the entire evaluation pass; instead we record
``None`` for that metric and surface a per-metric ``errors`` field so callers
(and the CLI) can flag the issue without losing the metrics that did work.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

from captioning.evaluation.bleu import corpus_bleu_breakdown
from captioning.evaluation.cider import MIN_SAMPLES_FOR_CIDER, corpus_cider_score
from captioning.evaluation.meteor import corpus_meteor_score
from captioning.evaluation.rouge import corpus_rouge_l_score


@dataclass(frozen=True)
class MetricsReport:
    """Aggregate metric snapshot for one evaluation pass.

    Every metric is ``float | None`` — ``None`` means the metric was skipped
    (uninstalled, environment missing Java, too few samples for CIDEr, ...).
    The reason for skipping is in :attr:`errors` keyed by metric name.
    """

    n_examples: int
    bleu1: float | None = None
    bleu2: float | None = None
    bleu3: float | None = None
    bleu4: float | None = None
    rouge_l: float | None = None
    meteor: float | None = None
    cider: float | None = None
    errors: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable dict (``errors`` becomes a sub-object)."""
        return asdict(self)


def compute_all_metrics(
    predictions: Sequence[str],
    references: Sequence[Sequence[str]],
    *,
    include_meteor: bool = True,
    include_cider: bool = True,
) -> MetricsReport:
    """Compute every available metric on a single ``(preds, refs)`` corpus.

    Args:
        predictions: One generated caption per example.
        references: One *list* of reference captions per example.
        include_meteor: Set False to skip METEOR (avoids the JVM spawn —
            helpful in CI where Java isn't installed).
        include_cider: Set False to skip CIDEr (avoids the warning when
            running on tiny corpora; the runner also auto-skips below
            ``MIN_SAMPLES_FOR_CIDER``).

    Returns:
        A :class:`MetricsReport` with every field populated by a corpus
        metric or recorded as failed in ``errors``.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions ({len(predictions)}) and references "
            f"({len(references)}) must have the same length"
        )

    errors: dict[str, str] = {}
    bleu1 = bleu2 = bleu3 = bleu4 = None
    rouge_l = meteor = cider = None

    try:
        bleu = corpus_bleu_breakdown(predictions, references)
        bleu1, bleu2, bleu3, bleu4 = bleu.bleu1, bleu.bleu2, bleu.bleu3, bleu.bleu4
    except Exception as e:  # — surface, don't crash the run
        errors["bleu"] = repr(e)

    try:
        rouge_l = corpus_rouge_l_score(predictions, references)
    except Exception as e:
        errors["rouge_l"] = repr(e)

    if include_meteor:
        try:
            meteor = corpus_meteor_score(predictions, references)
        except Exception as e:
            errors["meteor"] = repr(e)

    if include_cider:
        if len(predictions) < MIN_SAMPLES_FOR_CIDER:
            errors["cider"] = (
                f"skipped: needs >= {MIN_SAMPLES_FOR_CIDER} examples, " f"got {len(predictions)}"
            )
        else:
            try:
                cider = corpus_cider_score(predictions, references)
            except Exception as e:
                errors["cider"] = repr(e)

    return MetricsReport(
        n_examples=len(predictions),
        bleu1=bleu1,
        bleu2=bleu2,
        bleu3=bleu3,
        bleu4=bleu4,
        rouge_l=rouge_l,
        meteor=meteor,
        cider=cider,
        errors=errors,
    )
