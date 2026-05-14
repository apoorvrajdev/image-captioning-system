"""Per-sample inspection utilities for diagnosing weak captions.

The aggregate corpus metric tells you *how bad* the model is; this module
tells you *why*. For each (image, prediction, reference-set) triple it
records per-sample BLEU-4, sentence-level ROUGE-L, the prediction length,
the longest repeated token run, and whether the prediction is empty after
stripping sentinels.

Three failure modes the evaluation pass is trying to surface:
    * **Generic captions** — high BLEU-1, low BLEU-4 (n-gram trickle out).
    * **Repetition** — large ``repeat_run`` value.
    * **Early stopping** — ``length_tokens`` far below reference median.

Output JSONL is intentionally flat (one line per sample) so it can be loaded
with ``pandas.read_json(..., lines=True)`` or grep'd from the shell. The
runner that uses this module writes one such file per evaluation pass
alongside ``metrics.json`` for the same run.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path

from captioning.evaluation.tokenization import strip_sentinels


@dataclass(frozen=True)
class SampleDiagnostics:
    """Inspectable record for one (image, prediction, reference-set) triple."""

    image: str
    prediction: str
    references: list[str]
    length_tokens: int
    longest_repeat_run: int
    sentence_bleu4: float | None
    sentence_rouge_l: float | None
    flags: list[str]


def _longest_repeat_run(tokens: Sequence[str]) -> int:
    """Return the longest run of immediately-repeated tokens.

    Example: ``["a", "a", "a", "dog"]`` -> ``3``. Used to flag the classic
    transformer-decoder collapse where the same token is emitted on every step.
    """
    if not tokens:
        return 0
    best = current = 1
    for prev, cur in pairwise(tokens):
        current = current + 1 if cur == prev else 1
        best = max(best, current)
    return best


def _sentence_bleu4(prediction: str, references: Sequence[str]) -> float | None:
    """Sentence-level BLEU-4 via sacrebleu's effective-order smoothing."""
    try:
        import sacrebleu
    except ImportError:
        return None
    if not references or not prediction:
        return None
    scorer = sacrebleu.metrics.BLEU(effective_order=True, max_ngram_order=4)
    return float(scorer.sentence_score(prediction, list(references)).score)


def _sentence_rouge_l(prediction: str, references: Sequence[str]) -> float | None:
    """Best-of-references sentence-level ROUGE-L F-measure (0-100 scale)."""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return None
    valid_refs = [r for r in references if r]
    if not valid_refs or not prediction:
        return None
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    best = max(scorer.score(r, prediction)["rougeL"].fmeasure for r in valid_refs)
    return float(100.0 * best)


def diagnose_sample(
    image: str,
    prediction: str,
    references: Sequence[str],
) -> SampleDiagnostics:
    """Return :class:`SampleDiagnostics` for one prediction-vs-references row."""
    pred_clean = strip_sentinels(prediction)
    ref_clean = [strip_sentinels(r) for r in references if r]
    tokens = pred_clean.split()

    flags: list[str] = []
    if not pred_clean:
        flags.append("empty")
    if len(tokens) <= 2:
        flags.append("very_short")
    repeat = _longest_repeat_run(tokens)
    if repeat >= 3:
        flags.append("repetitive")
    if ref_clean and tokens and len(tokens) < min(len(r.split()) for r in ref_clean) // 2:
        flags.append("under_length")

    return SampleDiagnostics(
        image=image,
        prediction=pred_clean,
        references=ref_clean,
        length_tokens=len(tokens),
        longest_repeat_run=repeat,
        sentence_bleu4=_sentence_bleu4(pred_clean, ref_clean),
        sentence_rouge_l=_sentence_rouge_l(pred_clean, ref_clean),
        flags=flags,
    )


def diagnose_many(
    images: Sequence[str],
    predictions: Sequence[str],
    references: Sequence[Sequence[str]],
) -> list[SampleDiagnostics]:
    """Vectorised :func:`diagnose_sample` over parallel sequences."""
    if not (len(images) == len(predictions) == len(references)):
        raise ValueError(
            "images, predictions, references must be the same length: "
            f"got {len(images)} / {len(predictions)} / {len(references)}"
        )
    return [
        diagnose_sample(img, pred, refs)
        for img, pred, refs in zip(images, predictions, references, strict=True)
    ]


def write_diagnostics_jsonl(
    diagnostics: Iterable[SampleDiagnostics],
    path: str | Path,
) -> None:
    """Write one JSON object per line — pandas/jq friendly.

    Args:
        diagnostics: An iterable of :class:`SampleDiagnostics` (typically the
            output of :func:`diagnose_many`).
        path: Destination file. Parent directory is created if needed.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for d in diagnostics:
            f.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")


def format_diagnostic_row(d: SampleDiagnostics) -> str:
    """Return a one-line human-readable summary — used by the CLI tail print."""
    bleu = f"BLEU4={d.sentence_bleu4:5.1f}" if d.sentence_bleu4 is not None else "BLEU4=    n/a"
    rouge = f"R-L={d.sentence_rouge_l:5.1f}" if d.sentence_rouge_l is not None else "R-L=    n/a"
    flagstr = ",".join(d.flags) if d.flags else "-"
    return (
        f"{Path(d.image).name:35s}  "
        f"{bleu}  {rouge}  len={d.length_tokens:>2}  repeat={d.longest_repeat_run:>2}  "
        f"flags={flagstr}\n  pred: {d.prediction}\n  ref : {d.references[0] if d.references else ''}"
    )
