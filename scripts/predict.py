"""CLI single-image inference.

Usage:
    python -m scripts.predict \\
        --config configs/base.yaml \\
        --weights models/v1.0.0/model.h5 \\
        --tokenizer-dir models/v1.0.0 \\
        --image path/to/photo.jpg
"""

from __future__ import annotations

from pathlib import Path

import click

from captioning.config import load_config
from captioning.inference import CaptionPredictor
from captioning.utils import configure_logging, get_logger

log = get_logger(__name__)


@click.command()
@click.option(
    "--config", "config_path", required=True, type=click.Path(exists=True, path_type=Path)
)
@click.option("--weights", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--tokenizer-dir", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--image", required=True, type=click.Path(exists=True, path_type=Path))
def main(config_path: Path, weights: Path, tokenizer_dir: Path, image: Path) -> None:
    """Generate a caption for one image."""
    configure_logging()
    config = load_config(config_path)

    predictor = CaptionPredictor.from_artifacts(
        weights_path=weights,
        tokenizer_dir=tokenizer_dir,
        config=config,
    )
    predictor.warmup()
    caption = predictor.predict_path(image)
    click.echo(caption)


if __name__ == "__main__":
    main()
