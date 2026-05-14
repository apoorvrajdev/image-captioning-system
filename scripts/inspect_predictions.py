"""Per-sample inspection — print N validation predictions vs. ground truth.

Usage:
    python -m scripts.inspect_predictions \\
        --config configs/base.yaml \\
        --weights models/v1.0.0/model.h5 \\
        --tokenizer-dir models/v1.0.0 \\
        --n-samples 25

This script answers the diagnostic question: *when the model is wrong, **how**
is it wrong?* Each row shows the image filename, the predicted caption, one
reference caption, sentence-level BLEU-4 / ROUGE-L, the prediction length,
the longest repeated-token run, and a set of failure flags
(``empty`` / ``very_short`` / ``repetitive`` / ``under_length``).

Output also lands as ``diagnostics.jsonl`` so the same data can be loaded
into pandas / DuckDB for ad-hoc grouping.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import cast

import click

from captioning.config import load_config
from captioning.data import load_coco_annotations, make_image_level_splits
from captioning.evaluation import (
    diagnose_many,
    format_diagnostic_row,
    write_diagnostics_jsonl,
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
@click.option("--n-samples", type=int, default=25, help="Number of random val samples to inspect.")
@click.option(
    "--decode-strategy",
    type=click.Choice(["greedy", "beam"]),
    default=None,
    help="Override decode strategy for this inspection.",
)
@click.option("--beam-width", type=int, default=None)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional path to write diagnostics.jsonl. Defaults to printing only.",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="RNG seed for sample selection (defaults to config.train.seed).",
)
def main(
    config_path: Path,
    weights: Path,
    tokenizer_dir: Path,
    n_samples: int,
    decode_strategy: str | None,
    beam_width: int | None,
    output_path: Path | None,
    seed: int | None,
) -> None:
    """Sample N val images, run inference, and print prediction diagnostics."""
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
    refs_by_image: dict[str, list[str]] = {}
    for img, cap in zip(val_imgs, val_caps, strict=True):
        refs_by_image.setdefault(img, []).append(cap)

    rng = random.Random(seed if seed is not None else config.train.seed)
    picks = rng.sample(sorted(refs_by_image.keys()), k=min(n_samples, len(refs_by_image)))

    effective_strategy = decode_strategy or config.serve.decode_strategy
    effective_beam_width = beam_width if beam_width is not None else config.serve.beam_width
    predictor = CaptionPredictor.from_artifacts(
        weights_path=weights,
        tokenizer_dir=tokenizer_dir,
        config=config,
        decode_strategy=cast("DecodeStrategy", effective_strategy),
        beam_width=effective_beam_width,
    )
    predictor.warmup()

    predictions = [predictor.predict_path(p) for p in picks]
    references = [refs_by_image[p] for p in picks]
    diagnostics = diagnose_many(picks, predictions, references)

    for d in diagnostics:
        click.echo(format_diagnostic_row(d))
        click.echo("-" * 80)

    if output_path is not None:
        write_diagnostics_jsonl(diagnostics, output_path)
        click.echo(f"Wrote diagnostics: {output_path}")


if __name__ == "__main__":
    main()
