"""Evaluate a trained model on the COCO validation split.

Usage:
    python -m scripts.evaluate \\
        --config configs/base.yaml \\
        --weights models/v1.0.0/model.h5 \\
        --tokenizer-dir models/v1.0.0 \\
        --report docs/results/v1.0.0.md \\
        --max-samples 500
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from captioning.config import load_config
from captioning.data import load_coco_annotations, make_image_level_splits
from captioning.evaluation import corpus_bleu_score
from captioning.inference import CaptionPredictor
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
    "--report",
    "report_path",
    default=None,
    type=click.Path(path_type=Path),
    help="Optional path to write a Markdown report.",
)
@click.option(
    "--max-samples",
    default=500,
    type=int,
    help="Cap on validation examples (full val takes hours on CPU).",
)
def main(
    config_path: Path,
    weights: Path,
    tokenizer_dir: Path,
    report_path: Path | None,
    max_samples: int,
) -> None:
    """Compute corpus BLEU-4 on the val split and (optionally) write a report."""
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

    predictor = CaptionPredictor.from_artifacts(
        weights_path=weights, tokenizer_dir=tokenizer_dir, config=config
    )
    predictor.warmup()

    predictions: list[str] = []
    references: list[list[str]] = []
    for path in image_paths:
        predictions.append(predictor.predict_path(path))
        references.append(refs_by_image[path])

    bleu = corpus_bleu_score(predictions, references)
    log.info("evaluation_done", bleu4=bleu, n=len(predictions))
    click.echo(f"BLEU-4: {bleu:.2f}  (n={len(predictions)})")

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            f"# Evaluation v1\n\n"
            f"- BLEU-4: **{bleu:.2f}**\n"
            f"- Examples: {len(predictions)}\n"
            f"- Weights: `{weights}`\n",
            encoding="utf-8",
        )
        json.dump(
            {"bleu4": bleu, "n": len(predictions)},
            (report_path.with_suffix(".json")).open("w", encoding="utf-8"),
            indent=2,
        )


if __name__ == "__main__":
    main()
