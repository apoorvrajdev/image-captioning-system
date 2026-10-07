"""Export the Phase 3 dashboard data from committed results.

The SPA's comparison dashboard reads one static JSON file, imported at build
time (ADR-021). This module builds it from two committed sources and nothing
else:

- the cross-run quality summary, ``results/<comparison_id>/comparison.json``
  (TASK-013, ``docs/EVAL_METHODOLOGY.md`` § 8);
- every latency run, ``results/<prefix><model_id>-<decoding>-<device>/latency.json``
  (TASK-015, § 9).

Values are copied verbatim, never recomputed or rounded. The only hand-written
content is each model's display name and the caveat notes, which restate
§§ 8 and 9. The sources are checked before anything is exported: quality and
latency must share the slice, every latency run must use the same protocol,
inputs, settings and timing, every summary must recompute from its raw
samples, and a model's Hub id and revision must agree across all its runs.
Any disagreement raises :class:`DashboardExportError` naming the run.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError

from captioning.evaluation.comparison import METRIC_KEYS, METRIC_LABELS
from captioning.evaluation.latency import summarize

DASHBOARD_SCHEMA_VERSION = 1
DEFAULT_COMPARISON_ID = "phase3-comparison"
DEFAULT_LATENCY_PREFIX = "phase3-latency-"
EXPORT_COMMAND = "python -m scripts.export_dashboard_data"

# Names shown in the dashboard, keyed by the § 8.1 model id.
DISPLAY_NAMES = {
    "blip-base": "BLIP-base",
    "git-base-coco": "GIT-base-coco",
    "inceptionv3-transformer-stabilized": "CNN + Transformer (InceptionV3)",
    "vit-gpt2": "ViT-GPT2",
}
# How each adapter handles a batch (§ 9.5): one generate() call for the whole
# batch, or one single-image prediction after another.
BATCH_MODES = {"CNNCaptioner": "sequential", "HFCaptioner": "batched"}

QUALITY_NOTES = (
    "Scores use the slice's stored references (about {rpi:.2f} per image), not the "
    "five-reference COCO setup, so they can't be compared with published COCO results "
    "(docs/EVAL_METHODOLOGY.md § 8.8).",
    'Rows of kind "reference" are committed CNN + Transformer runs made before the comparison '
    "harness, so they record no revision (null). The greedy one is reproduced exactly by a "
    "harness run at the pinned revision (docs/EVAL_METHODOLOGY.md § 8.8). The beam-search row "
    "is a labelled reference and isn't compared with the greedy baselines "
    "(docs/EVAL_METHODOLOGY.md § 8.4).",
    "Models are listed by model id. The order is not a ranking.",
)
LATENCY_NOTES = (
    "Values are seconds per call, and one call captions one whole batch, so a batch-8 value is "
    "the time for eight images, not per image (docs/EVAL_METHODOLOGY.md § 9.4).",
    'Runs with batch_mode "sequential" (the CNN + Transformer) caption a batch one image at a '
    "time: their batch-8 values are eight single-image calls in a row, not batched inference, "
    "and aren't like-for-like with the batched Hugging Face values (docs/EVAL_METHODOLOGY.md "
    "§ 9.5).",
    "Each device's runs come from a different host, operating system and framework build (see "
    "each run's environment). Figures compare only within one environment and device; together "
    "they are not a controlled CPU-versus-GPU comparison (docs/EVAL_METHODOLOGY.md §§ 9.5, 9.10).",
    "load_seconds is the cold-start cost on that host: captioner construction plus load(), "
    "including framework imports and any download. It is never part of a sample "
    "(docs/EVAL_METHODOLOGY.md § 9.1).",
    "Latency says nothing about caption quality (docs/EVAL_METHODOLOGY.md § 9.4).",
)


class DashboardExportError(ValueError):
    """The committed results can't be exported: a file is missing, malformed or inconsistent."""


class _Artefact(BaseModel):
    # Strict: a malformed value (a number stored as a string) is refused, not converted.
    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())


# A copied number keeps its JSON type: an int is never widened to a float.
_Number = int | float


class _ComparisonSlice(_Artefact):
    fingerprint_sha256: str
    images: int
    references: int
    references_per_image: _Number


class _ComparisonRow(_Artefact):
    run_id: str
    kind: str
    model_id: str
    backend: str | None
    captioner: str | None
    hub_repo: str | None
    revision: str | None
    weights_path: str
    decode_strategy: str
    decode_settings: dict[str, Any]
    n_samples: int
    metrics: dict[str, _Number | None]
    metric_errors: dict[str, str]


class _Comparison(_Artefact):
    protocol: str | None
    normalisation: str | None
    slice: _ComparisonSlice
    caveat: str
    metric_keys: list[str]
    rows: list[_ComparisonRow]


class _LatencySlice(_Artefact):
    source: str
    fingerprint_sha256: str
    images: int
    references: int


class _LatencyInputs(_Artefact):
    slice: _LatencySlice
    images: list[str]


class _LatencySettings(_Artefact):
    num_images: int
    batch_sizes: list[int]
    warmup_passes: int
    measured_passes: int


class _Timing(_Artefact):
    clock: str
    sample: str
    load: str


class _Summary(_Artefact):
    count: int
    mean: _Number
    median: _Number
    min: _Number
    max: _Number


class _Batch(_Artefact):
    batch_size: int
    calls_per_pass: int
    samples_seconds: list[_Number]
    summary_seconds: _Summary


class _Runtime(_Artefact):
    python: str
    platform: str
    packages: dict[str, str | None]
    tensorflow_gpus: list[str] | None


class _Latency(_Artefact):
    protocol: str
    model_id: str
    backend: str
    captioner: str
    hub_repo: str | None
    revision: str | None
    decode_strategy: str
    decode_settings: dict[str, Any]
    device: str
    environment: str
    inputs: _LatencyInputs
    settings: _LatencySettings
    timing: _Timing
    load_seconds: _Number
    batches: list[_Batch]
    seed: int
    runtime: _Runtime


_ModelT = TypeVar("_ModelT", bound=BaseModel)


def find_latency_runs(results_root: str | Path, prefix: str = DEFAULT_LATENCY_PREFIX) -> list[Path]:
    """Return every ``<results_root>/<prefix>*/`` latency run directory, sorted by name."""
    return sorted(p for p in Path(results_root).glob(f"{prefix}*") if p.is_dir())


def build_dashboard_data(
    results_root: str | Path,
    *,
    comparison_id: str = DEFAULT_COMPARISON_ID,
    latency_prefix: str = DEFAULT_LATENCY_PREFIX,
) -> dict[str, Any]:
    """Check the committed Phase 3 results and return the dashboard data.

    Args:
        results_root: The ``results/`` directory.
        comparison_id: The quality summary's directory under ``results_root``.
        latency_prefix: Every directory under ``results_root`` whose name starts
            with this is read as a latency run.

    Raises:
        DashboardExportError: If a source is missing, malformed or inconsistent.
    """
    results_root = Path(results_root)
    comparison = _read(results_root / comparison_id / "comparison.json", _Comparison, comparison_id)
    if comparison.metric_keys != list(METRIC_KEYS):
        raise DashboardExportError(
            f"{comparison_id}: metric_keys {comparison.metric_keys} differ from {list(METRIC_KEYS)}"
        )
    latency_dirs = find_latency_runs(results_root, latency_prefix)
    if not latency_dirs:
        raise DashboardExportError(
            f"no latency runs found: no {latency_prefix}* directory under {results_root.as_posix()}"
        )
    runs = [(d.name, _read(d / "latency.json", _Latency, d.name)) for d in latency_dirs]
    base_id, base = runs[0]
    for run_id, run in runs:
        _check_latency_run(run_id, run, latency_prefix)
        for label, value, expected in (
            ("protocol", run.protocol, base.protocol),
            ("inputs", run.inputs, base.inputs),
            ("settings", run.settings, base.settings),
            ("timing", run.timing, base.timing),
            ("seed", run.seed, base.seed),
        ):
            if value != expected:
                raise DashboardExportError(f"{run_id}: {label} doesn't match {base_id}")
    latency_slice = base.inputs.slice
    for label, value, expected in (
        (
            "slice fingerprint",
            latency_slice.fingerprint_sha256,
            comparison.slice.fingerprint_sha256,
        ),
        ("slice image count", latency_slice.images, comparison.slice.images),
        ("slice reference count", latency_slice.references, comparison.slice.references),
    ):
        if value != expected:
            raise DashboardExportError(
                f"{base_id}: {label} {value} differs from {comparison_id} ({expected})"
            )
    for run_id in [row.run_id for row in comparison.rows]:
        if not (results_root / run_id).is_dir():
            raise DashboardExportError(
                f"{comparison_id}: source run {run_id} has no directory under "
                f"{results_root.as_posix()}"
            )

    model_ids = sorted({row.model_id for row in comparison.rows} | {r.model_id for _, r in runs})
    unnamed = [model_id for model_id in model_ids if model_id not in DISPLAY_NAMES]
    if unnamed:
        raise DashboardExportError(
            f"no display name for model(s) {unnamed}; add it to DISPLAY_NAMES"
        )

    s = comparison.slice
    return {
        "schema_version": DASHBOARD_SCHEMA_VERSION,
        "generated_by": EXPORT_COMMAND,
        "slice": {
            "description": (
                f"{s.images} COCO train2017 images, in the order of {latency_slice.source}, each "
                f"scored against its stored references: {s.references} in total, about "
                f"{s.references_per_image:.2f} per image. The latency runs time the first "
                f"{base.settings.num_images} of these images."
            ),
            "source": latency_slice.source,
            "fingerprint_sha256": s.fingerprint_sha256,
            "images": s.images,
            "references": s.references,
            "references_per_image": s.references_per_image,
        },
        "overlap_caveat": comparison.caveat,
        "quality": {
            "protocol": comparison.protocol,
            "normalisation": comparison.normalisation,
            "summary_run_id": comparison_id,
            "metrics": [
                {"key": key, "label": label}
                for key, label in zip(METRIC_KEYS, METRIC_LABELS, strict=True)
            ],
            "notes": [note.format(rpi=s.references_per_image) for note in QUALITY_NOTES],
        },
        "latency": {
            "protocol": base.protocol,
            "unit": "seconds per call; one call captions one batch",
            "statistics": list(_Summary.model_fields),
            "settings": base.settings.model_dump(),
            "timing": base.timing.model_dump(),
            "notes": list(LATENCY_NOTES),
        },
        "models": [_model_entry(model_id, comparison, runs) for model_id in model_ids],
    }


def render_dashboard_json(data: dict[str, Any]) -> str:
    """Serialise :func:`build_dashboard_data` output exactly as it is committed."""
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def _check_latency_run(run_id: str, run: _Latency, prefix: str) -> None:
    expected_id = f"{prefix}{run.model_id}-{run.decode_strategy}-{run.device}"
    if run_id != expected_id:
        raise DashboardExportError(
            f"{run_id}: latency.json describes {expected_id} (model, decoding and device)"
        )
    if run.captioner not in BATCH_MODES:
        raise DashboardExportError(f"{run_id}: unknown captioner {run.captioner!r}")
    batch_sizes = [batch.batch_size for batch in run.batches]
    if batch_sizes != run.settings.batch_sizes:
        raise DashboardExportError(
            f"{run_id}: batches {batch_sizes} differ from settings.batch_sizes "
            f"{run.settings.batch_sizes}"
        )
    for batch in run.batches:
        expected_count = batch.calls_per_pass * run.settings.measured_passes
        if len(batch.samples_seconds) != expected_count:
            raise DashboardExportError(
                f"{run_id}: batch size {batch.batch_size} has {len(batch.samples_seconds)} "
                f"samples, expected {expected_count}"
            )
        if summarize(batch.samples_seconds) != batch.summary_seconds.model_dump():
            raise DashboardExportError(
                f"{run_id}: batch size {batch.batch_size} summary_seconds doesn't match its "
                "samples_seconds"
            )


def _model_entry(
    model_id: str, comparison: _Comparison, runs: Sequence[tuple[str, _Latency]]
) -> dict[str, Any]:
    rows = [row for row in comparison.rows if row.model_id == model_id]
    latency = sorted(
        ((run_id, run) for run_id, run in runs if run.model_id == model_id),
        key=lambda item: (item[1].device, item[1].decode_strategy, item[0]),
    )
    identity = {
        field: _agreed(
            model_id,
            field,
            [(row.run_id, getattr(row, field)) for row in rows]
            + [(run_id, getattr(run, field)) for run_id, run in latency],
        )
        for field in ("backend", "hub_repo", "revision")
    }
    quality = [
        {
            "run_id": row.run_id,
            "kind": row.kind,
            "revision": row.revision,
            "decode_strategy": row.decode_strategy,
            "decode_settings": row.decode_settings,
            "n_samples": row.n_samples,
            "metrics": row.metrics,
            "metric_errors": row.metric_errors,
        }
        for row in rows
    ]
    latency_entries = [
        {
            "run_id": run_id,
            "device": run.device,
            "environment": run.environment,
            "decode_strategy": run.decode_strategy,
            "batch_mode": BATCH_MODES[run.captioner],
            "load_seconds": run.load_seconds,
            "batches": [
                {
                    "batch_size": batch.batch_size,
                    "calls_per_pass": batch.calls_per_pass,
                    "summary_seconds": batch.summary_seconds.model_dump(),
                }
                for batch in run.batches
            ],
        }
        for run_id, run in latency
    ]
    return {
        "model_id": model_id,
        "display_name": DISPLAY_NAMES[model_id],
        **identity,
        "source_run_ids": [entry["run_id"] for entry in quality + latency_entries],
        "quality": quality,
        "latency": latency_entries,
    }


def _agreed(model_id: str, field: str, values: list[tuple[str, str | None]]) -> str:
    """Return the one value every run that records ``field`` gives for ``model_id``."""
    recorded = {value for _, value in values if value is not None}
    if not recorded:
        raise DashboardExportError(f"{model_id}: no run records its {field}")
    if len(recorded) > 1:
        detail = ", ".join(f"{run_id}={value}" for run_id, value in values if value is not None)
        raise DashboardExportError(f"{model_id}: runs disagree on {field}: {detail}")
    return recorded.pop()


def _read(path: Path, model: type[_ModelT], run_id: str) -> _ModelT:
    if not path.is_file():
        raise DashboardExportError(f"{run_id}: missing {path.name}")
    try:
        return model.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (ValueError, ValidationError) as exc:
        detail = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        raise DashboardExportError(f"{run_id}: {path.name} is malformed: {detail}") from exc
