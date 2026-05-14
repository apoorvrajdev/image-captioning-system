"""Benchmark-ready run artefacts.

Every evaluation pass writes a consistent set of files under
``<run_root>/<run_id>/`` so Phase 3 cross-model comparisons can join them
without bespoke parsing per model:

    metrics.json            — :class:`MetricsReport` dumped via dataclass-asdict
    predictions.jsonl       — one row per (image, prediction, references)
    diagnostics.jsonl       — one :class:`SampleDiagnostics` per row
    run_meta.json           — model id, decode strategy, n_samples, timestamp
    report.md               — Markdown summary humans actually read

A "run" is one (model, decode_strategy, dataset_slice) tuple. ``run_id`` is
a free-form string — the CLI defaults to a timestamp; comparison code groups
by ``model_id`` to plot bars across models.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from captioning.evaluation.inspection import SampleDiagnostics, write_diagnostics_jsonl
from captioning.evaluation.runner import MetricsReport


@dataclass(frozen=True)
class RunMeta:
    """Per-evaluation-run metadata persisted next to metrics."""

    model_id: str
    decode_strategy: str
    weights_path: str
    tokenizer_dir: str
    n_samples: int
    max_length: int
    beam_width: int | None = None
    length_penalty: float | None = None
    repetition_penalty: float | None = None
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "decode_strategy": self.decode_strategy,
            "weights_path": self.weights_path,
            "tokenizer_dir": self.tokenizer_dir,
            "n_samples": self.n_samples,
            "max_length": self.max_length,
            "beam_width": self.beam_width,
            "length_penalty": self.length_penalty,
            "repetition_penalty": self.repetition_penalty,
            "timestamp_utc": self.timestamp_utc,
        }


def write_run_artifacts(
    run_dir: str | Path,
    *,
    metrics: MetricsReport,
    meta: RunMeta,
    images: list[str],
    predictions: list[str],
    references: list[list[str]],
    diagnostics: list[SampleDiagnostics],
) -> Path:
    """Write every benchmark artefact to ``run_dir`` and return the directory.

    Idempotent over a clean ``run_dir``; overwrites existing files inside.
    """
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "metrics.json").write_text(json.dumps(metrics.to_dict(), indent=2), encoding="utf-8")
    (out / "run_meta.json").write_text(json.dumps(meta.to_dict(), indent=2), encoding="utf-8")

    with (out / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for img, pred, refs in zip(images, predictions, references, strict=True):
            row = {"image": img, "prediction": pred, "references": list(refs)}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_diagnostics_jsonl(diagnostics, out / "diagnostics.jsonl")
    (out / "report.md").write_text(_render_report_markdown(meta, metrics), encoding="utf-8")
    return out


def _render_report_markdown(meta: RunMeta, m: MetricsReport) -> str:
    """Render the human-facing Markdown summary of a single run."""

    def fmt(v: float | None) -> str:
        return "n/a" if v is None else f"{v:.2f}"

    lines = [
        f"# Evaluation run — {meta.model_id}",
        "",
        f"- Decode strategy: `{meta.decode_strategy}`",
        f"- Weights: `{meta.weights_path}`",
        f"- Tokenizer dir: `{meta.tokenizer_dir}`",
        f"- Samples: **{meta.n_samples}**",
        f"- Timestamp (UTC): {meta.timestamp_utc}",
    ]
    if meta.beam_width is not None:
        lines.append(f"- Beam width: {meta.beam_width}")
    if meta.length_penalty is not None:
        lines.append(f"- Length penalty: {meta.length_penalty}")
    if meta.repetition_penalty is not None:
        lines.append(f"- Repetition penalty: {meta.repetition_penalty}")
    lines += [
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| BLEU-1 | {fmt(m.bleu1)} |",
        f"| BLEU-2 | {fmt(m.bleu2)} |",
        f"| BLEU-3 | {fmt(m.bleu3)} |",
        f"| BLEU-4 | {fmt(m.bleu4)} |",
        f"| ROUGE-L | {fmt(m.rouge_l)} |",
        f"| METEOR | {fmt(m.meteor)} |",
        f"| CIDEr | {fmt(m.cider)} |",
    ]
    if m.errors:
        lines += ["", "## Skipped or failed metrics", ""]
        for name, err in m.errors.items():
            lines.append(f"- `{name}`: {err}")
    return "\n".join(lines) + "\n"
