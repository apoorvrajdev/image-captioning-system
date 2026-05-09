"""Generate development-only model artifacts so the FastAPI backend can boot.

Why this script exists:
    The Phase 2 backend lifespan loads weights + tokenizer from
    ``models/v1.0.0/``. Until Phase 1 training has been run end-to-end on
    COCO, those files don't exist and ``uvicorn`` fails on startup with
    ``FileNotFoundError``. This script produces a *valid* but
    *not meaningfully trained* set of artefacts so:

      * the entire backend pipeline (lifespan, /healthz, /v1/captions,
        multipart upload, predictor wiring) can be exercised;
      * mypy/ruff/pytest stay green;
      * a recruiter reviewing the repo can run ``uvicorn`` and hit the API.

Captions returned by the bootstrapped model will be *gibberish* — every
weight is initialised by Keras's default initialiser and never trained.
That's deliberate and clearly documented; the goal is to verify the
serving system, not produce real predictions.

Usage::

    python -m scripts.bootstrap_dev_artifacts \\
        --config configs/base.yaml \\
        --output-dir models/v1.0.0

The script is idempotent — running it twice overwrites the previous
artefacts. To replace dev artefacts with real Phase 1 outputs, run
``scripts/train.py`` and copy ``model.h5`` + ``vocab.pkl`` into the same
directory.
"""

from __future__ import annotations

from pathlib import Path

import click

from captioning.config import load_config
from captioning.models.factory import build_caption_model
from captioning.preprocessing.tokenizer import CaptionTokenizer
from captioning.utils import configure_logging, get_logger

log = get_logger(__name__)

# A tiny synthetic corpus. Wrapped in [start] ... [end] to mirror exactly the
# pre-processed format the real training pipeline produces in cell 4. The
# vocabulary that comes out of fitting on this is small (~50 tokens), but
# that's fine: the model's vocab_size is taken from the fitted tokenizer at
# build time, so weights and decode tables stay in lockstep.
_DEV_CORPUS: list[str] = [
    "[start] a man riding a surfboard on a wave [end]",
    "[start] a woman holding a small dog in her arms [end]",
    "[start] a group of people standing on a beach [end]",
    "[start] a cat sitting on top of a wooden table [end]",
    "[start] a plate of food on a wooden table [end]",
    "[start] a red bus driving down a city street [end]",
    "[start] a child kicking a soccer ball in a park [end]",
    "[start] two birds sitting on a tree branch [end]",
    "[start] a kitchen with a stove and a refrigerator [end]",
    "[start] a person standing in front of a mountain [end]",
]


@click.command()
@click.option(
    "--config",
    "config_path",
    default=Path("configs/base.yaml"),
    show_default=True,
    type=click.Path(exists=True, path_type=Path),
    help="App config YAML. Architecture hyperparameters are read from `model.*`.",
)
@click.option(
    "--output-dir",
    default=Path("models/v1.0.0"),
    show_default=True,
    type=click.Path(path_type=Path),
    help="Directory that will contain model.h5, vocab.pkl, vocab.json.",
)
def main(config_path: Path, output_dir: Path) -> None:
    """Create model.h5 + vocab.pkl + vocab.json under ``output-dir``."""
    configure_logging()
    config = load_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_filename = config.train.weights_filename
    weights_path = output_dir / weights_filename

    log.info("bootstrap_starting", output_dir=str(output_dir))

    # 1. Fit a tiny tokenizer on the synthetic corpus and save it.
    tokenizer = CaptionTokenizer(
        vocab_size=config.model.vocabulary_size,
        max_length=config.model.max_length,
    )
    tokenizer.fit(_DEV_CORPUS)
    tokenizer.save(output_dir)
    log.info(
        "tokenizer_saved",
        directory=str(output_dir),
        vocabulary_size=tokenizer.vocabulary_size,
    )

    # 2. Build the model with the *fitted* vocab size so the weights file
    #    matches the tokenizer that will be loaded next to it. Augmentation
    #    is left at its default (enabled) so the variable tree matches what
    #    a real Phase 1 ``model.fit`` produces — the predictor builds with
    #    the same defaults on load.
    model = build_caption_model(config, vocab_size=tokenizer.vocabulary_size)

    # 3. Force a forward pass so all variables are created before save. The
    #    sequence of calls mirrors ``CaptionPredictor._dummy_pass`` exactly,
    #    keeping save/load symmetric.
    import tensorflow as tf

    dummy_img = tf.zeros((1, 299, 299, 3), dtype=tf.float32)
    dummy_caps = tf.zeros((1, config.model.max_length), dtype=tf.int64)
    img_embed = model.cnn_model(dummy_img)
    encoded = model.encoder(img_embed, training=False)
    _ = model.decoder(
        dummy_caps[:, :-1],
        encoded,
        training=False,
        mask=tf.cast(dummy_caps[:, 1:] != 0, tf.int32),
    )
    if getattr(model, "image_aug", None) is not None:
        _ = model.image_aug(dummy_img, training=False)

    # 4. Mark the parent Model as built so HDF5 save/load round-trips. Real
    #    Phase 1 weights satisfy this implicitly via ``model.fit``; the
    #    bootstrap doesn't fit, so we set the flag explicitly. Predictor's
    #    ``_dummy_pass`` does the symmetric thing on load.
    model.built = True

    # 5. Save randomly-initialised weights. The file is structurally identical
    #    to a real Phase 1 checkpoint; only the values inside are untrained.
    model.save_weights(str(weights_path))
    log.info(
        "weights_saved",
        path=str(weights_path),
        warning="weights are randomly initialised; outputs will be gibberish",
    )

    click.echo(
        "\nDevelopment artefacts written:\n"
        f"  weights : {weights_path}\n"
        f"  vocab   : {output_dir / 'vocab.pkl'}\n"
        f"  vocab   : {output_dir / 'vocab.json'}\n"
        "\nThese are SMOKE-TEST artefacts only. Replace with real Phase 1 "
        "outputs before drawing any inference about model quality."
    )


if __name__ == "__main__":
    main()
