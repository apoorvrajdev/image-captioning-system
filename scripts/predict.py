"""CLI single-image inference.

Usage:
    # Greedy (default — same as the IEEE notebook)
    python -m scripts.predict \\
        --config configs/base.yaml \\
        --weights models/v1.0.0/model.h5 \\
        --tokenizer-dir models/v1.0.0 \\
        --image path/to/photo.jpg

    # Beam search with explicit parameters
    python -m scripts.predict ... --decode-strategy beam --beam-width 4 \\
        --length-penalty 0.7 --repetition-penalty 1.1 --no-repeat-ngram-size 3
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import click

from captioning.config import load_config
from captioning.inference import CaptionPredictor
from captioning.inference.predictor import DecodeStrategy
from captioning.utils import configure_logging, get_logger

log = get_logger(__name__)


@click.command()
@click.option(
    "--config", "config_path", required=True, type=click.Path(exists=True, path_type=Path)
)
@click.option("--weights", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--tokenizer-dir", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--image", required=True, type=click.Path(exists=True, path_type=Path))
@click.option(
    "--decode-strategy",
    type=click.Choice(["greedy", "beam"]),
    default=None,
    help="Override config.serve.decode_strategy for this run.",
)
@click.option("--beam-width", type=int, default=None)
@click.option("--length-penalty", type=float, default=None)
@click.option("--repetition-penalty", type=float, default=None)
@click.option("--no-repeat-ngram-size", type=int, default=None)
def main(
    config_path: Path,
    weights: Path,
    tokenizer_dir: Path,
    image: Path,
    decode_strategy: str | None,
    beam_width: int | None,
    length_penalty: float | None,
    repetition_penalty: float | None,
    no_repeat_ngram_size: int | None,
) -> None:
    """Generate a caption for one image."""
    configure_logging()
    config = load_config(config_path)

    predictor = CaptionPredictor.from_artifacts(
        weights_path=weights,
        tokenizer_dir=tokenizer_dir,
        config=config,
        decode_strategy=cast("DecodeStrategy | None", decode_strategy),
        beam_width=beam_width,
        length_penalty=length_penalty,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )
    predictor.warmup()
    caption = predictor.predict_path(image)
    click.echo(caption)


if __name__ == "__main__":
    main()
