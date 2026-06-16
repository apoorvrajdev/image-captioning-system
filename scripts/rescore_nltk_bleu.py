"""One-off diagnostic: rescore committed predictions under varying BLEU regimes.

This script isolates the two evaluation-methodology axes that separate this
repo's metrics from the IEEE notebook's reported BLEU-4 ~24:

  * Axis A  — aggregation + smoothing (sacrebleu corpus vs NLTK smoothed
              sentence-BLEU). A previous rescore showed this is a wash under
              defensible smoothers when scored against the same references.
  * Axis B  — reference count. The committed predictions.jsonl carries only
              ~1.46 references/image (most have a single reference), not COCO's
              canonical 5. This script joins the full 5-reference set from the
              official annotations file and rescores against it.

When ``--coco-annotations`` is omitted the script reproduces the original
~1.46-ref behaviour, so the two reference-count regimes can be compared
side-by-side in the same session.

The scripts that gate the Kaggle retraining run are split in two — this one
(BLEU) and ``scripts/categorize_predictions.py`` (blinded qualitative read) —
and are run in SEPARATE turns so the BLEU number cannot bias the qualitative
categorization. The two scripts share no code and no state.

Usage
-----
    # 5-ref gating test (the real test):
    python -m scripts.rescore_nltk_bleu \
        --predictions-path results/stabilized-beam-w4-lp07-rp12/predictions.jsonl \
        --coco-annotations /path/to/captions_train2017.json

    # 1.46-ref reproduction (omit --coco-annotations):
    python -m scripts.rescore_nltk_bleu \
        --predictions-path results/stabilized-beam-w4-lp07-rp12/predictions.jsonl

PRE-REGISTERED BLEU PREDICTION
------------------------------
HYPOTHESIS: Reference-count is the dominant remaining axis of the IEEE eval
methodology gap (Axis A — aggregation+smoothing — was already shown to be a
wash by the previous rescore under defensible smoothers).

PREDICTION: 5-ref sacrebleu corpus BLEU-4 will land >= 18.

DECISION RULE (BLEU-only):
  >= 18  -> DOMINANT. Methodology (reference count) dominates the IEEE gap.
  14-18  -> MAJOR-BUT-PARTIAL. Methodology is a major but partial factor.
  <= 13  -> MINOR. Methodology contributes ~3 points at most. Checkpoint
            genuinely underperforms IEEE.

SECONDARY: 5-ref NLTK method1 should land within ~1 point of 5-ref sacrebleu
corpus. If it diverges by >3 points, Axis A is not a wash under 5 refs after
all and the multi-axis analysis above needs revision.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import click
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

# Sentinels the training pipeline wraps captions in (mirrors
# captioning.preprocessing.caption). Inlined so this one-off stays standalone
# and TF-free (fast to run as a Kaggle cell).
START_TOKEN = "[start]"
END_TOKEN = "[end]"

# Caption normalisation identical to captioning.preprocessing.caption
# .preprocess_caption MINUS the sentinel wrap: lowercase, strip punctuation,
# collapse whitespace. Applied to raw COCO captions so the 5-ref set lands in
# the SAME token space as the predictions and the stored 1.46-ref set —
# otherwise the reference-count axis would be contaminated by a tokenisation
# mismatch.
import re  # noqa: E402  (kept next to the patterns it owns)

_PUNCTUATION_RE = re.compile(r"[^\w\s]")
_WHITESPACE_RE = re.compile(r"\s+")

# Cumulative BLEU-n weight vectors (uniform over the first n orders).
_CUMULATIVE_WEIGHTS = {
    1: (1.0, 0.0, 0.0, 0.0),
    2: (0.5, 0.5, 0.0, 0.0),
    3: (1 / 3, 1 / 3, 1 / 3, 0.0),
    4: (0.25, 0.25, 0.25, 0.25),
}

_SMOOTHERS = {
    "method0": SmoothingFunction().method0,
    "method1": SmoothingFunction().method1,
    "method4": SmoothingFunction().method4,
    "method7": SmoothingFunction().method7,
}


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace (no sentinels)."""
    if not text:
        return ""
    text = text.lower()
    text = _PUNCTUATION_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def _strip_sentinels(caption: str) -> str:
    """Remove [start]/[end], lowercase, collapse whitespace."""
    if not caption:
        return ""
    cleaned = caption.replace(START_TOKEN, " ").replace(END_TOKEN, " ")
    return _normalize(cleaned)


def _image_id(image_path: str) -> int:
    """COCO image_id from a .../train2017/000000530117.jpg path."""
    stem = Path(image_path).stem
    return int(stem)  # raises ValueError on a non-numeric stem (malformed path)


def _load_predictions(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_coco_refs(path: Path) -> dict[int, list[str]]:
    """Build {image_id: [raw captions...]} from captions_train2017.json."""
    data = json.loads(path.read_text(encoding="utf-8"))
    refs: dict[int, list[str]] = {}
    for ann in data["annotations"]:
        refs.setdefault(int(ann["image_id"]), []).append(ann["caption"])
    return refs


def _refs_by_slot(references: Sequence[Sequence[str]]) -> list[list[str]]:
    """Ragged per-example references -> sacrebleu per-slot layout."""
    max_refs = max((len(r) for r in references), default=0)
    return [[refs[i] if i < len(refs) else "" for refs in references] for i in range(max_refs)]


def _sacrebleu_breakdown(preds: list[str], references: list[list[str]]) -> dict[int, float]:
    """sacrebleu corpus BLEU-1..4, same config as captioning.evaluation.bleu."""
    import sacrebleu

    refs_by_slot = _refs_by_slot(references)
    out: dict[int, float] = {}
    for n in (1, 2, 3, 4):
        scorer = sacrebleu.metrics.BLEU(max_ngram_order=n, effective_order=True)
        out[n] = float(scorer.corpus_score(preds, refs_by_slot).score)
    return out


def _nltk_macro_breakdown(
    hyps: list[list[str]], refs: list[list[list[str]]], smoother
) -> dict[int, float]:
    """Macro-averaged sentence BLEU-1..4 (0-100) for a given smoother."""
    sums = {n: 0.0 for n in (1, 2, 3, 4)}
    for hyp, ref_list in zip(hyps, refs, strict=True):
        if not hyp:
            continue
        for n in (1, 2, 3, 4):
            sums[n] += sentence_bleu(
                ref_list, hyp, weights=_CUMULATIVE_WEIGHTS[n], smoothing_function=smoother
            )
    count = len(hyps) or 1
    return {n: 100.0 * sums[n] / count for n in (1, 2, 3, 4)}


def _band(bleu4: float) -> str:
    """Map 5-ref sacrebleu corpus BLEU-4 to the pre-registered band."""
    if bleu4 >= 18.0:
        return "DOMINANT"
    if bleu4 >= 14.0:
        return "MAJOR-BUT-PARTIAL"
    if bleu4 <= 13.0:
        return "MINOR"
    return "BOUNDARY-13-14-REVIEW"  # 13 < x < 14 — left undefined by the spec


@click.command()
@click.option(
    "--predictions-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("results/stabilized-beam-w4-lp07-rp12/predictions.jsonl"),
    help="predictions.jsonl from a scripts.evaluate run.",
)
@click.option(
    "--coco-annotations",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="captions_train2017.json. When given, scores against all 5 COCO refs; "
    "when omitted, reproduces the ~1.46-ref behaviour.",
)
@click.option(
    "--smoother",
    type=click.Choice(list(_SMOOTHERS)),
    default="method1",
    help="Primary NLTK smoother for the headline table (method4 always also shown).",
)
def main(predictions_path: Path, coco_annotations: Path | None, smoother: str) -> None:
    """Rescore predictions; in 5-ref mode emit the pre-registered band."""
    rows = _load_predictions(predictions_path)
    five_ref = coco_annotations is not None

    # ---- Build the active reference set ------------------------------------
    preds: list[str] = []
    refs_sacre: list[list[str]] = []  # normalised strings, per example
    if coco_annotations is not None:
        coco = _load_coco_refs(coco_annotations)
        missing: list[int] = []
        for rec in rows:
            iid = _image_id(rec["image"])
            if iid not in coco:
                missing.append(iid)
        if missing:
            raise click.ClickException(
                f"{len(missing)} prediction image_id(s) absent from "
                f"{coco_annotations}. First 5: {missing[:5]}. "
                "Refusing to fall back to single-ref scoring."
            )
        for rec in rows:
            iid = _image_id(rec["image"])
            preds.append(_normalize(rec["prediction"]))
            refs_sacre.append([_normalize(c) for c in coco[iid]])
    else:
        for rec in rows:
            preds.append(_normalize(rec["prediction"]))
            refs_sacre.append([_strip_sentinels(r) for r in rec["references"]])

    # NLTK works on token lists.
    hyps_tok = [p.split() for p in preds]
    refs_tok = [[r.split() for r in ref_list if r] for ref_list in refs_sacre]

    # ---- Reference-count stats (A2) ----------------------------------------
    counts = [len(r) for r in refs_sacre]
    n = len(counts)
    ref_stats = {
        "n_examples": n,
        "mean": round(sum(counts) / n, 4) if n else 0.0,
        "min": min(counts) if counts else 0,
        "max": max(counts) if counts else 0,
        "n_lt_5": sum(1 for c in counts if c < 5),
    }

    # ---- Metrics -----------------------------------------------------------
    new_sacre = _sacrebleu_breakdown(preds, refs_sacre)
    new_primary = _nltk_macro_breakdown(hyps_tok, refs_tok, _SMOOTHERS[smoother])
    new_method4 = _nltk_macro_breakdown(hyps_tok, refs_tok, _SMOOTHERS["method4"])

    metrics_path = predictions_path.parent / "metrics.json"
    committed = (
        json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    )
    committed_bleu = {n_: committed.get(f"bleu{n_}", float("nan")) for n_ in (1, 2, 3, 4)}

    regime = "5-ref" if five_ref else "1.46-ref"

    # ---- Headline four-column table (NO method7) ---------------------------
    click.echo(f"Predictions   : {predictions_path}")
    click.echo(
        f"Reference set : {regime}  "
        f"(mean {ref_stats['mean']}/image, min {ref_stats['min']}, "
        f"max {ref_stats['max']}, {ref_stats['n_lt_5']}/{n} have <5 refs)"
    )
    click.echo("")
    header = (
        f"  {'metric':<8}{'committed sacre':>16}{f'new sacre ({regime})':>20}"
        f"{f'NLTK {smoother}':>16}{'NLTK method4':>16}"
    )
    click.echo(header)
    for n_ in (1, 2, 3, 4):
        click.echo(
            f"  BLEU-{n_:<3}{committed_bleu[n_]:>16.2f}{new_sacre[n_]:>20.2f}"
            f"{new_primary[n_]:>16.2f}{new_method4[n_]:>16.2f}"
        )
    click.echo("")

    # ---- Secondary check + band (5-ref only) -------------------------------
    if five_ref:
        delta = new_primary[4] - new_sacre[4]
        axis_a_wash = abs(delta) <= 3.0
        band = _band(new_sacre[4])
        click.echo(f"SECONDARY CHECK: NLTK method1 BLEU-4 - sacrebleu corpus BLEU-4 = {delta:+.2f}")
        click.echo(
            "  -> Axis A is a wash under 5 refs (|delta| <= 3)."
            if axis_a_wash
            else "  -> Axis A is NOT a wash under 5 refs (|delta| > 3); revise the multi-axis analysis."
        )
        click.echo("")
        click.echo(
            f"PRE-REGISTERED BAND (5-ref sacrebleu corpus BLEU-4 = {new_sacre[4]:.2f}): {band}"
        )
        click.echo("")

        out = {
            "predictions_path": str(predictions_path),
            "coco_annotations": str(coco_annotations),
            "ref_stats": ref_stats,
            "committed_1p46ref_sacrebleu": {f"bleu{k}": committed_bleu[k] for k in (1, 2, 3, 4)},
            "new_5ref": {
                "sacrebleu_corpus": {f"bleu{k}": new_sacre[k] for k in (1, 2, 3, 4)},
                f"nltk_{smoother}": {f"bleu{k}": new_primary[k] for k in (1, 2, 3, 4)},
                "nltk_method4": {f"bleu{k}": new_method4[k] for k in (1, 2, 3, 4)},
            },
            "band": band,
            "band_basis": "5ref_sacrebleu_corpus_bleu4",
            "secondary_check": {
                "method1_vs_corpus_bleu4_delta": round(delta, 4),
                "axis_a_wash_under_5ref": axis_a_wash,
            },
        }
        out_path = predictions_path.parent / "metrics_5ref.json"
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        click.echo(f"Wrote: {out_path}")
    else:
        click.echo(
            "1.46-ref mode: band + metrics_5ref.json skipped "
            "(pass --coco-annotations to run the gating test)."
        )

    # ---- Diagnostic: smoother sensitivity (NOT headline) -------------------
    click.echo("")
    click.echo("DIAGNOSTIC (smoother sensitivity — NOT the headline; method7 is known to inflate):")
    click.echo(f"  {'smoother':<10}{'BLEU-4':>10}")
    for name in ("method0", "method1", "method4", "method7"):
        b4 = _nltk_macro_breakdown(hyps_tok, refs_tok, _SMOOTHERS[name])[4]
        click.echo(f"  {name:<10}{b4:>10.2f}")


if __name__ == "__main__":
    main()
