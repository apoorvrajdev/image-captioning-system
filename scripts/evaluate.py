"""Evaluate a trained model on the COCO validation split.

Usage:
    # Full benchmark-ready evaluation (recommended) — writes
    # results/<run_id>/{metrics.json, predictions.jsonl, diagnostics.jsonl, ...}
    python -m scripts.evaluate \\
        --config configs/base.yaml \\
        --weights models/v1.0.0/model.h5 \\
        --tokenizer-dir models/v1.0.0 \\
        --results-root results \\
        --max-samples 500

    # Optional: produce a single Markdown report at a chosen path
    python -m scripts.evaluate ... --report docs/results/v1.0.0.md

What this script produces (per run):
    metrics.json        — corpus BLEU-1..4, ROUGE-L, METEOR, CIDEr
    predictions.jsonl   — image / prediction / references for downstream tools
    diagnostics.jsonl   — per-sample length / repetition / sentence BLEU flags
    run_meta.json       — model id, decode strategy, beam width, timestamp
    report.md           — human-readable summary

Phase 3 benchmark code joins multiple ``results/<run_id>/`` directories to
plot BLEU-4 / CIDEr / latency across models.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import click

from captioning.config import load_config
from captioning.data import load_coco_annotations, make_image_level_splits
from captioning.evaluation import (
    RunMeta,
    compute_all_metrics,
    diagnose_many,
    write_run_artifacts,
)
from captioning.inference import CaptionPredictor
from captioning.inference.predictor import DecodeStrategy
from captioning.preprocessing import preprocess_caption
from captioning.utils import configure_logging, get_logger, set_global_seed

log = get_logger(__name__)


@click.command()
@click.option(
    "--config", "config_path", required=True, type=click.Path(exists=True, path_type=Path)
)
@click.option("--weights", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--tokenizer-dir", required=True, type=click.Path(exists=True, path_type=Path))
@click.option(
    "--results-root",
    type=click.Path(path_type=Path),
    default=Path("results"),
    help="Parent directory for the per-run sub-folder (results/<run_id>/).",
)
@click.option(
    "--run-id",
    type=str,
    default=None,
    help="Sub-folder name under --results-root. Defaults to a UTC timestamp.",
)
@click.option(
    "--model-id",
    type=str,
    default="inceptionv3-transformer-v1",
    help="Identifier used by Phase 3 cross-model joining of metrics.",
)
@click.option(
    "--decode-strategy",
    type=click.Choice(["greedy", "beam"]),
    default=None,
    help="Override config.serve.decode_strategy for this run.",
)
@click.option("--beam-width", type=int, default=None, help="Beam width (only used with beam).")
@click.option("--length-penalty", type=float, default=None)
@click.option("--repetition-penalty", type=float, default=None)
@click.option(
    "--report",
    "report_path",
    default=None,
    type=click.Path(path_type=Path),
    help="Optional path to an additional human-readable Markdown report.",
)
@click.option(
    "--max-samples",
    default=500,
    type=int,
    help="Cap on validation examples (full val takes hours on CPU).",
)
@click.option(
    "--skip-meteor",
    is_flag=True,
    default=False,
    help="Skip METEOR (avoids needing Java).",
)
@click.option(
    "--skip-cider",
    is_flag=True,
    default=False,
    help="Skip CIDEr.",
)
def main(  # — CLI option count is unavoidable
    config_path: Path,
    weights: Path,
    tokenizer_dir: Path,
    results_root: Path,
    run_id: str | None,
    model_id: str,
    decode_strategy: str | None,
    beam_width: int | None,
    length_penalty: float | None,
    repetition_penalty: float | None,
    report_path: Path | None,
    max_samples: int,
    skip_meteor: bool,
    skip_cider: bool,
) -> None:
    """Evaluate the model on the val split and write benchmark artefacts."""
    configure_logging()
    config = load_config(config_path)
    set_global_seed(config.train.seed)

    df = load_coco_annotations(
        base_path=config.data.base_path,
        annotations_filename=config.data.annotations_filename,
        images_subdir=config.data.images_subdir,
        sample_size=config.data.sample_size,
        seed=config.train.seed,
        caption_preprocessor=preprocess_caption,
    )
    _, _, val_imgs, val_caps = make_image_level_splits(
        df, train_fraction=config.data.train_val_split, seed=config.train.seed
    )

    # Group references by image so we get the COCO 5-references-per-image format.
    refs_by_image: dict[str, list[str]] = {}
    for img, cap in zip(val_imgs, val_caps, strict=True):
        refs_by_image.setdefault(img, []).append(cap)
    image_paths = list(refs_by_image.keys())[:max_samples]

    effective_strategy = decode_strategy or config.serve.decode_strategy
    effective_beam_width = beam_width if beam_width is not None else config.serve.beam_width
    effective_length_penalty = (
        length_penalty if length_penalty is not None else config.serve.length_penalty
    )
    effective_repetition_penalty = (
        repetition_penalty if repetition_penalty is not None else config.serve.repetition_penalty
    )

    predictor = CaptionPredictor.from_artifacts(
        weights_path=weights,
        tokenizer_dir=tokenizer_dir,
        config=config,
        decode_strategy=cast("DecodeStrategy", effective_strategy),
        beam_width=effective_beam_width,
        length_penalty=effective_length_penalty,
        repetition_penalty=effective_repetition_penalty,
    )
    predictor.warmup()

    predictions: list[str] = []
    references: list[list[str]] = []
    for path in image_paths:
        predictions.append(predictor.predict_path(path))
        references.append(refs_by_image[path])

    metrics = compute_all_metrics(
        predictions,
        references,
        include_meteor=not skip_meteor,
        include_cider=not skip_cider,
    )
    diagnostics = diagnose_many(image_paths, predictions, references)

    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(results_root) / run_id
    meta = RunMeta(
        model_id=model_id,
        decode_strategy=effective_strategy,
        weights_path=str(weights),
        tokenizer_dir=str(tokenizer_dir),
        n_samples=len(predictions),
        max_length=config.model.max_length,
        beam_width=effective_beam_width if effective_strategy == "beam" else None,
        length_penalty=effective_length_penalty if effective_strategy == "beam" else None,
        repetition_penalty=effective_repetition_penalty,
    )
    write_run_artifacts(
        run_dir,
        metrics=metrics,
        meta=meta,
        images=image_paths,
        predictions=predictions,
        references=references,
        diagnostics=diagnostics,
    )

    log.info(
        "evaluation_done",
        run_dir=str(run_dir),
        n=metrics.n_examples,
        bleu4=metrics.bleu4,
        rouge_l=metrics.rouge_l,
        meteor=metrics.meteor,
        cider=metrics.cider,
    )
    click.echo(f"Run directory: {run_dir}")
    _echo_metric("BLEU-1", metrics.bleu1)
    _echo_metric("BLEU-2", metrics.bleu2)
    _echo_metric("BLEU-3", metrics.bleu3)
    _echo_metric("BLEU-4", metrics.bleu4)
    _echo_metric("ROUGE-L", metrics.rouge_l)
    _echo_metric("METEOR", metrics.meteor)
    _echo_metric("CIDEr", metrics.cider)
    if metrics.errors:
        click.echo(f"Skipped/failed: {sorted(metrics.errors)}")

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            (run_dir / "report.md").read_text(encoding="utf-8"), encoding="utf-8"
        )


def _echo_metric(name: str, value: float | None) -> None:
    if value is None:
        click.echo(f"{name}: n/a")
    else:
        click.echo(f"{name}: {value:.2f}")


if __name__ == "__main__":
    main()
