"""Train the IEEE InceptionV3+Transformer captioning model.

Usage:
    python -m scripts.train --config configs/base.yaml
    python -m scripts.train --config configs/base.yaml --output-dir models/v1.0.0

The script orchestrates the same pipeline as the notebook, but each step is
imported from the modular package — making it the canonical example of how
the package is meant to be composed.
"""

from __future__ import annotations

from pathlib import Path

import click

from captioning.config import load_config
from captioning.data import (
    build_train_pipeline,
    build_val_pipeline,
    load_coco_annotations,
    make_image_level_splits,
)
from captioning.models import build_caption_model
from captioning.preprocessing import CaptionTokenizer, preprocess_caption
from captioning.training import Trainer
from captioning.utils import configure_logging, get_logger, set_global_seed

log = get_logger(__name__)


@click.command()
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="YAML config file (e.g. configs/base.yaml).",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default="outputs/runs/latest",
    help="Where to save weights, vocab, and history.",
)
def main(config_path: Path, output_dir: Path) -> None:
    """Run the full training pipeline end-to-end."""
    configure_logging()
    config = load_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(config.train.seed)
    log.info("config_loaded", path=str(config_path), output_dir=str(output_dir))

    # 1. Load + preprocess COCO captions ------------------------------------
    df = load_coco_annotations(
        base_path=config.data.base_path,
        annotations_filename=config.data.annotations_filename,
        images_subdir=config.data.images_subdir,
        sample_size=config.data.sample_size,
        seed=config.train.seed,
        caption_preprocessor=preprocess_caption,
    )

    # 2. Fit and persist the tokenizer --------------------------------------
    tokenizer = CaptionTokenizer(
        vocab_size=config.model.vocabulary_size,
        max_length=config.model.max_length,
    )
    tokenizer.fit(df["caption"])
    tokenizer.save(output_dir)

    # 3. Image-level train/val split ----------------------------------------
    train_imgs, train_caps, val_imgs, val_caps = make_image_level_splits(
        df, train_fraction=config.data.train_val_split, seed=config.train.seed
    )

    # 4. tf.data pipelines ---------------------------------------------------
    train_ds = build_train_pipeline(
        train_imgs,
        train_caps,
        tokenizer,
        batch_size=config.train.batch_size,
        buffer_size=config.train.buffer_size,
    )
    val_ds = build_val_pipeline(
        val_imgs,
        val_caps,
        tokenizer,
        batch_size=config.train.batch_size,
        buffer_size=config.train.buffer_size,
    )

    # 5. Build, compile, fit -------------------------------------------------
    model = build_caption_model(config, vocab_size=tokenizer.vocabulary_size)
    trainer = Trainer(model, config)
    trainer.fit(train_ds, val_ds, output_dir=output_dir)

    # 6. Save final weights to the canonical filename ------------------------
    final_weights = output_dir / config.train.weights_filename
    model.save_weights(str(final_weights))
    log.info("training_done", weights=str(final_weights))


if __name__ == "__main__":
    main()
