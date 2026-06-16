"""Blinded qualitative categorization of caption predictions (Part B).

This is the qualitative half of the two-part evaluation-methodology gate. It
tests CIDEr-style image-specificity: does the checkpoint produce captions that
meaningfully constrain the image, or generic category labels? It runs in a
SEPARATE Claude Code turn from the BLEU rescore so the BLEU number cannot bias
the categorization. By construction this script:

  * does NOT import scripts/rescore_nltk_bleu.py,
  * does NOT read results/.../metrics_5ref.json (or any BLEU output),
  * shares no state with Part A.

The blinding is structural: when the sample is drawn from qualitative.jsonl,
the per-sample metric fields (sentence_bleu4, sentence_rouge_l, flags, ...) are
dropped; only ``image`` and ``prediction`` are carried forward, then the full
5-reference set is joined from the official COCO annotations.

Workflow (two modes)
--------------------
  PREPARE  (default, no --categories):
      Select the blinded sample, join 5 COCO refs, PRINT each sample as a
      numbered block, and write a pending --output (category/justification
      = null). The operator/agent then categorizes each sample BY JUDGMENT
      using the rubric below.
  FINALIZE (--categories PATH):
      Read a JSONL of {sample_id, category, justification}, validate every
      category against the four allowed values (invented categories raise),
      enforce the <=25-word justification limit, merge, write the final
      --output, and print category COUNTS (never naked percentages).

CATEGORIZATION RUBRIC
---------------------
Each prediction is assigned EXACTLY ONE category:

  SPECIFIC-CORRECT
    Caption identifies the main subject correctly AND includes at least one
    distinguishing attribute that meaningfully constrains the image — color,
    count, named action, spatial relation, or a named secondary object.
    Example: "a woman riding a brown horse on a beach."

  GENERIC-CORRECT
    Caption identifies the main subject correctly but lacks any distinguishing
    attribute — the same caption could describe many photos of the category.
    Example: "a person on a beach."

  PARTIALLY-CORRECT
    Identifies at least one element correctly and gets at least one other
    element wrong (wrong color, wrong count, wrong action, wrong secondary
    object). Example for an image of two skiers: "a man riding skis."

  INCORRECT
    Misidentifies the main subject, is incoherent, or describes a scene
    clearly absent from all 5 references.

COMBINED DECISION RULE (BLEU verdict from Part A + SPECIFIC-CORRECT count from
Part B):

  BLEU DOMINANT (>=18) AND SPECIFIC-CORRECT >= 12/30
    -> Strong evidence for the metric-parity reframe. Don't retrain.
       Pivot to Stage 7 reframe in the plan.

  BLEU MAJOR-BUT-PARTIAL (14-18) AND SPECIFIC-CORRECT >= 15/30
    -> Qualitative is strong enough to ship without retraining.
       Reframe with a "BLEU underestimates this checkpoint" note.

  BLEU MAJOR-BUT-PARTIAL (14-18) AND SPECIFIC-CORRECT < 10/30
    -> Qualitative confirms BLEU. Retrain (Kaggle Stage 1).

  BLEU MINOR (<=13)
    -> Retrain regardless of qualitative. Metric gap is too large to argue
       around with 30 samples.

  Any combination not covered above (e.g. DOMINANT with SPECIFIC-CORRECT <12,
  or MAJOR-BUT-PARTIAL with SPECIFIC-CORRECT 10-14)
    -> Flag for human review. Do not auto-decide.

Usage
-----
    # PREPARE the blinded worklist:
    python -m scripts.categorize_predictions \
        --coco-annotations /path/to/captions_train2017.json

    # FINALIZE after judgment:
    python -m scripts.categorize_predictions \
        --coco-annotations /path/to/captions_train2017.json \
        --categories results/stabilized-beam-w4-lp07-rp12/categories.jsonl
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import click

# The four categories are fixed. Renaming or adding to this set (e.g.
# "ALMOST-SPECIFIC") is forbidden — finalize mode raises on any other value.
ALLOWED_CATEGORIES = ("SPECIFIC-CORRECT", "GENERIC-CORRECT", "PARTIALLY-CORRECT", "INCORRECT")
MAX_JUSTIFICATION_WORDS = 25

_RESULTS_DIR = Path("results/stabilized-beam-w4-lp07-rp12")


def _image_id(image_path: str) -> int:
    return int(Path(image_path).stem)


def _load_coco_refs(path: Path) -> dict[int, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    refs: dict[int, list[str]] = {}
    for ann in data["annotations"]:
        refs.setdefault(int(ann["image_id"]), []).append(ann["caption"])
    return refs


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _select_sample(predictions_path: Path, sample_size: int, seed: int) -> list[dict]:
    """Return [{image, prediction}] for the blinded sample (metrics dropped)."""
    qualitative = _RESULTS_DIR / "qualitative.jsonl"
    if qualitative.exists():
        rows = _read_jsonl(qualitative)
        if len(rows) >= sample_size:
            # Blinding: carry ONLY image + prediction; drop sentence_bleu4,
            # sentence_rouge_l, flags, length_tokens, etc.
            return [
                {"image": r["image"], "prediction": r["prediction"]} for r in rows[:sample_size]
            ]
    rows = _read_jsonl(predictions_path)
    picks = random.Random(seed).sample(rows, k=min(sample_size, len(rows)))
    return [{"image": r["image"], "prediction": r["prediction"]} for r in picks]


def _join_refs(sample: list[dict], coco: dict[int, list[str]]) -> list[dict]:
    """Attach the full COCO ref list; raise (no fallback) on missing ids."""
    missing = [_image_id(s["image"]) for s in sample if _image_id(s["image"]) not in coco]
    if missing:
        raise click.ClickException(
            f"{len(missing)} sample image_id(s) absent from the annotations file. "
            f"First 5: {missing[:5]}. Refusing to proceed with partial references."
        )
    out: list[dict] = []
    for s in sample:
        iid = _image_id(s["image"])
        out.append({"sample_id": str(iid), "prediction": s["prediction"], "refs": coco[iid]})
    return out


@click.command()
@click.option(
    "--predictions-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_RESULTS_DIR / "predictions.jsonl",
    help="predictions.jsonl (fallback sample source if qualitative.jsonl is absent).",
)
@click.option(
    "--coco-annotations",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="captions_train2017.json — full 5-reference set per image.",
)
@click.option("--sample-size", type=int, default=30, help="Number of samples to categorize.")
@click.option("--seed", type=int, default=42, help="RNG seed (only used for the fallback sampler).")
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    default=_RESULTS_DIR / "qualitative_categorized.jsonl",
    help="Where the categorized rows are written.",
)
@click.option(
    "--categories",
    "categories_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="JSONL of {sample_id, category, justification}. Omit for PREPARE mode.",
)
def main(
    predictions_path: Path,
    coco_annotations: Path,
    sample_size: int,
    seed: int,
    output_path: Path,
    categories_path: Path | None,
) -> None:
    """Prepare a blinded sample (default) or finalize a categorized sample."""
    coco = _load_coco_refs(coco_annotations)
    sample = _select_sample(predictions_path, sample_size, seed)
    joined = _join_refs(sample, coco)
    n = len(joined)

    if categories_path is None:
        # ---- PREPARE: print blinded worklist, write pending output ---------
        click.echo(f"Blinded worklist: {n} samples (predictions + 5 refs only; NO metrics shown).")
        click.echo("Categorize each per the rubric in this script's docstring, then re-run")
        click.echo("with --categories pointing at a JSONL of {sample_id, category, justification}.")
        click.echo("=" * 80)
        for i, row in enumerate(joined, start=1):
            click.echo(f"[{i:>2}] sample_id={row['sample_id']}")
            click.echo(f"     PRED: {row['prediction']}")
            for j, ref in enumerate(row["refs"], start=1):
                click.echo(f"     ref{j}: {ref}")
            click.echo("-" * 80)
        pending = [
            {
                "sample_id": r["sample_id"],
                "prediction": r["prediction"],
                "refs": r["refs"],
                "category": None,
                "justification": None,
            }
            for r in joined
        ]
        with output_path.open("w", encoding="utf-8") as f:
            for row in pending:
                f.write(json.dumps(row) + "\n")
        click.echo(f"Wrote pending worklist ({n} rows, categories null): {output_path}")
        return

    # ---- FINALIZE: validate, merge, write, count ---------------------------
    cat_rows = {r["sample_id"]: r for r in _read_jsonl(categories_path)}
    final: list[dict] = []
    for row in joined:
        sid = row["sample_id"]
        if sid not in cat_rows:
            raise click.ClickException(f"sample_id {sid} missing from {categories_path}.")
        category = cat_rows[sid]["category"]
        justification = cat_rows[sid].get("justification", "")
        if category not in ALLOWED_CATEGORIES:
            raise click.ClickException(
                f"sample_id {sid}: category {category!r} is not one of {ALLOWED_CATEGORIES}. "
                "The rubric's four categories may not be renamed or expanded."
            )
        if len(str(justification).split()) > MAX_JUSTIFICATION_WORDS:
            raise click.ClickException(
                f"sample_id {sid}: justification exceeds {MAX_JUSTIFICATION_WORDS} words."
            )
        final.append(
            {
                "sample_id": sid,
                "prediction": row["prediction"],
                "refs": row["refs"],
                "category": category,
                "justification": justification,
            }
        )

    with output_path.open("w", encoding="utf-8") as f:
        for row in final:
            f.write(json.dumps(row) + "\n")

    counts = Counter(r["category"] for r in final)
    click.echo(f"Categorized {n} samples -> {output_path}")
    click.echo("Counts:")
    click.echo("  " + ", ".join(f"{counts.get(c, 0)}/{n} {c}" for c in ALLOWED_CATEGORIES))
    specific = counts.get("SPECIFIC-CORRECT", 0)
    click.echo("")
    click.echo(f"SPECIFIC-CORRECT = {specific}/{n}  (input to the COMBINED DECISION RULE)")
    click.echo("Apply the COMBINED DECISION RULE in this script's docstring together with the")
    click.echo("Part A band. This script does NOT read the BLEU output — do not auto-decide here.")
    click.echo(
        f"Note: with N={n}, a proportion carries roughly +/-18% sampling margin; "
        "report counts, not point-estimate percentages."
    )


if __name__ == "__main__":
    main()
