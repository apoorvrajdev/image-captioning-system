"""Join Phase 3 run directories into one cross-model comparison summary.

Every compared run must score the same slice: the same image file names, in
the same order, with the same references (``docs/EVAL_METHODOLOGY.md`` § 8.2).
Identity is the slice fingerprint, recomputed from each run's
``predictions.jsonl`` and checked against what its ``comparison_meta.json``
records (fingerprint, counts, protocol, normalisation). Nothing is repaired:
any disagreement raises :class:`ComparisonError` naming the run.

Committed runs made before the comparison runner existed (the CNN +
Transformer rows of § 8.4) have no ``comparison_meta.json``. They are loaded
only when asked for explicitly, as labelled reference rows identified by the
same fingerprint. Metric values are copied verbatim from each ``metrics.json``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError

from captioning.evaluation.slice import load_eval_slice, slice_fingerprint

METRIC_KEYS = ("bleu1", "bleu2", "bleu3", "bleu4", "rouge_l", "meteor", "cider")
METRIC_LABELS = ("BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-L", "METEOR", "CIDEr")
OVERLAP_CAVEAT = (
    "The slice comes from COCO train2017. The CNN + Transformer held these images out of "
    "training; the Hugging Face baselines were fine-tuned on COCO training data (ViT-GPT2 "
    "possibly), so they may have seen them. These scores are not a held-out, like-for-like "
    "comparison (docs/EVAL_METHODOLOGY.md § 8.5)."
)


class ComparisonError(ValueError):
    """Runs can't be compared: a run is incomplete, malformed or on another slice."""


class _Artefact(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class _Metrics(_Artefact):
    n_examples: int
    bleu1: float | None
    bleu2: float | None
    bleu3: float | None
    bleu4: float | None
    rouge_l: float | None
    meteor: float | None
    cider: float | None
    errors: dict[str, str]


class _RunMeta(_Artefact):
    model_id: str
    decode_strategy: str
    weights_path: str
    tokenizer_dir: str
    n_samples: int
    max_length: int
    beam_width: int | None
    length_penalty: float | None
    repetition_penalty: float | None
    timestamp_utc: str


class _SliceMeta(_Artefact):
    source: str
    fingerprint_sha256: str
    images: int
    references: int


class _MetricFlags(_Artefact):
    meteor: bool
    cider: bool


class _ComparisonMeta(_Artefact):
    protocol: str
    model_id: str
    backend: str
    captioner: str
    hub_repo: str | None
    revision: str | None
    decode_settings: dict[str, Any]
    normalisation: str
    slice: _SliceMeta
    batch_size: int
    device: str | None
    seed: int
    metrics: _MetricFlags


@dataclass(frozen=True)
class RunRecord:
    """One validated run directory, ready to compare."""

    run_id: str
    metrics: _Metrics
    run_meta: _RunMeta
    comparison_meta: _ComparisonMeta | None  # None for a pre-harness reference run
    fingerprint: str
    images: int
    references: int


_ModelT = TypeVar("_ModelT", bound=BaseModel)


def load_run(run_dir: str | Path, *, reference: bool = False) -> RunRecord:
    """Read and check one run directory.

    Args:
        run_dir: A ``results/<run_id>/`` directory.
        reference: ``True`` for a committed run made before the comparison
            runner, which has no ``comparison_meta.json``.

    Raises:
        ComparisonError: If a file is missing or malformed, or the run's files
            disagree with each other about the slice.
    """
    run_dir = Path(run_dir)
    run_id = run_dir.name
    required = ["predictions.jsonl", "metrics.json", "run_meta.json"]
    if not reference:
        required.append("comparison_meta.json")
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise ComparisonError(f"{run_id}: missing {', '.join(missing)}")

    metrics = _parse(run_dir / "metrics.json", _Metrics, run_id)
    run_meta = _parse(run_dir / "run_meta.json", _RunMeta, run_id)
    meta = None if reference else _parse(run_dir / "comparison_meta.json", _ComparisonMeta, run_id)
    try:
        eval_slice = load_eval_slice(run_dir / "predictions.jsonl", ".")
    except ValueError as exc:  # includes JSON decoding errors
        raise ComparisonError(f"{run_id}: predictions.jsonl is malformed: {exc}") from exc

    images = len(eval_slice)
    references = sum(len(refs) for refs in eval_slice.references)
    fingerprint = slice_fingerprint(eval_slice)
    counts = {
        "run_meta.json n_samples": run_meta.n_samples,
        "metrics.json n_examples": metrics.n_examples,
    }
    if meta is not None:
        counts["comparison_meta.json slice.images"] = meta.slice.images
    for label, value in counts.items():
        if value != images:
            raise ComparisonError(
                f"{run_id}: {label} is {value}, but predictions.jsonl has {images} images"
            )
    if meta is not None:
        if meta.slice.references != references:
            raise ComparisonError(
                f"{run_id}: comparison_meta.json slice.references is {meta.slice.references}, "
                f"but predictions.jsonl has {references} references"
            )
        if meta.slice.fingerprint_sha256 != fingerprint:
            raise ComparisonError(
                f"{run_id}: comparison_meta.json records slice fingerprint "
                f"{meta.slice.fingerprint_sha256}, but predictions.jsonl gives {fingerprint}"
            )
        if meta.model_id != run_meta.model_id:
            raise ComparisonError(
                f"{run_id}: comparison_meta.json model_id {meta.model_id!r} "
                f"differs from run_meta.json ({run_meta.model_id!r})"
            )
    return RunRecord(run_id, metrics, run_meta, meta, fingerprint, images, references)


def build_summary(runs: Sequence[RunRecord]) -> dict[str, Any]:
    """Check that ``runs`` are comparable and return the comparison summary.

    Rows are sorted by model id, decoding and run id, so the summary doesn't
    depend on the order the runs were given in.

    Raises:
        ComparisonError: If no runs are given, two runs share a model and
            decoding, or any run differs in slice, protocol or normalisation.
    """
    if not runs:
        raise ComparisonError("no runs to compare")
    ordered = sorted(
        runs, key=lambda r: (r.run_meta.model_id, r.run_meta.decode_strategy, r.run_id)
    )
    base = ordered[0]
    seen: dict[tuple[str, str], str] = {}
    for run in ordered:
        key = (run.run_meta.model_id, run.run_meta.decode_strategy)
        if key in seen:
            raise ComparisonError(
                f"{run.run_id}: duplicates {seen[key]} (model {key[0]}, {key[1]} decoding)"
            )
        seen[key] = run.run_id
        for label, value, expected in (
            ("image count", run.images, base.images),
            ("reference count", run.references, base.references),
            ("slice fingerprint", run.fingerprint, base.fingerprint),
        ):
            if value != expected:
                raise ComparisonError(
                    f"{run.run_id}: {label} {value} differs from {base.run_id} ({expected})"
                )

    harness_runs = [r for r in ordered if r.comparison_meta is not None]
    for run in harness_runs[1:]:
        for label in ("protocol", "normalisation"):
            value = getattr(run.comparison_meta, label)
            expected = getattr(harness_runs[0].comparison_meta, label)
            if value != expected:
                raise ComparisonError(
                    f"{run.run_id}: {label} {value!r} differs from "
                    f"{harness_runs[0].run_id} ({expected!r})"
                )
    first_meta = harness_runs[0].comparison_meta if harness_runs else None

    return {
        "protocol": first_meta.protocol if first_meta else None,
        "normalisation": first_meta.normalisation if first_meta else None,
        "slice": {
            "fingerprint_sha256": base.fingerprint,
            "images": base.images,
            "references": base.references,
            "references_per_image": round(base.references / base.images, 2),
        },
        "caveat": OVERLAP_CAVEAT,
        "metric_keys": list(METRIC_KEYS),
        "rows": [_row(run) for run in ordered],
    }


def render_markdown(summary: dict[str, Any]) -> str:
    """Render the human-readable table for a :func:`build_summary` result."""
    s = summary["slice"]
    lines = [
        "# Phase 3 model comparison",
        "",
        f"- Protocol: {summary['protocol'] or 'not recorded (reference runs only)'}",
        f"- Slice: {s['images']} images, {s['references']} references "
        f"(about {s['references_per_image']:.2f} per image), "
        f"fingerprint `{s['fingerprint_sha256']}`",
        f"- Normalisation: {summary['normalisation'] or 'not recorded (reference runs only)'}",
        "",
        f"> {summary['caveat']}",
        "",
        "| Run | Model | Decoding | Revision | Samples | " + " | ".join(METRIC_LABELS) + " |",
        "|---|---|---|---|---|" + "---|" * len(METRIC_LABELS),
    ]
    for row in summary["rows"]:
        run = f"`{row['run_id']}`" + (" (reference)" if row["kind"] == "reference" else "")
        revision = f"`{row['revision'][:7]}`" if row["revision"] else "—"
        values = " | ".join(_fmt(row["metrics"][key]) for key in METRIC_KEYS)
        lines.append(
            f"| {run} | {row['model_id']} | {row['decode_strategy']} | {revision} | "
            f"{row['n_samples']} | {values} |"
        )
    lines += [
        "",
        "Values are rounded to two decimals; `comparison.json` holds the exact values "
        "from each run's `metrics.json`.",
        "",
    ]
    return "\n".join(lines)


def _row(run: RunRecord) -> dict[str, Any]:
    meta, run_meta = run.comparison_meta, run.run_meta
    decode_settings = (
        meta.decode_settings
        if meta is not None
        else {
            "beam_width": run_meta.beam_width,
            "length_penalty": run_meta.length_penalty,
            "repetition_penalty": run_meta.repetition_penalty,
            "max_length": run_meta.max_length,
        }
    )
    return {
        "run_id": run.run_id,
        "kind": "reference" if meta is None else "phase3",
        "model_id": run_meta.model_id,
        "backend": meta.backend if meta else None,
        "captioner": meta.captioner if meta else None,
        "hub_repo": meta.hub_repo if meta else None,
        "revision": meta.revision if meta else None,
        "weights_path": run_meta.weights_path,
        "decode_strategy": run_meta.decode_strategy,
        "decode_settings": decode_settings,
        "n_samples": run_meta.n_samples,
        "metrics": {key: getattr(run.metrics, key) for key in METRIC_KEYS},
        "metric_errors": run.metrics.errors,
    }


def _parse(path: Path, model: type[_ModelT], run_id: str) -> _ModelT:
    try:
        return model.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (ValueError, ValidationError) as exc:
        detail = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        raise ComparisonError(f"{run_id}: {path.name} is malformed: {detail}") from exc


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"
