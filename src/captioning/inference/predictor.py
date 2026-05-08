"""``CaptionPredictor`` — stateful, FastAPI-friendly inference singleton.

Why a class around the existing functions:
    * The FastAPI lifespan loads weights once at boot and reuses the same
      model across every request. A predictor object is the natural home for
      "loaded model + loaded tokenizer + decoded config".
    * Tests can construct one with stub objects without monkey-patching globals.
    * Phase 1b adds beam search; Phase 3 adds a model registry. Both extend
      this class, not the functional callsites.

Construction is *not* the same as readiness: ``CaptionPredictor.warmup()``
runs one inference on a dummy tensor so the first real request doesn't pay
TF's lazy graph-build cost (typically 2-5 seconds).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from captioning.config.schema import AppConfig
from captioning.inference.greedy import generate_caption_greedy
from captioning.inference.image_loader import load_image_from_path
from captioning.preprocessing.tokenizer import CaptionTokenizer
from captioning.utils.logging import get_logger

log = get_logger(__name__)


class CaptionPredictor:
    """Thin wrapper exposing ``predict_path`` / ``predict_tensor`` / ``warmup``."""

    def __init__(
        self,
        model,
        tokenizer: CaptionTokenizer,
        config: AppConfig,
        *,
        decode_strategy: Literal["greedy"] = "greedy",
    ) -> None:
        """Args:
        model: Loaded ``ImageCaptioningModel``. Caller is responsible for
            having called ``model.load_weights(...)`` already.
        tokenizer: Fitted ``CaptionTokenizer``.
        config: Validated ``AppConfig`` — ``model.max_length`` is consumed.
        decode_strategy: Phase 1 supports only ``"greedy"``. Phase 1b adds
            ``"beam"``; this argument is here so the signature is stable.
        """
        if decode_strategy != "greedy":
            raise NotImplementedError(
                f"Phase 1 supports decode_strategy='greedy' only, got {decode_strategy!r}"
            )
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.decode_strategy = decode_strategy

    @classmethod
    def from_artifacts(
        cls,
        weights_path: str | Path,
        tokenizer_dir: str | Path,
        config: AppConfig,
    ) -> CaptionPredictor:
        """Load weights and tokenizer from disk and return a ready predictor.

        Args:
            weights_path: Path to ``model.h5`` (notebook cell 30 saved this).
            tokenizer_dir: Directory containing ``vocab.pkl`` (and ``vocab.json``).
            config: Validated ``AppConfig``. ``model.max_length`` and
                ``model.vocabulary_size`` must match the trained weights.

        Returns:
            A ``CaptionPredictor`` ready for inference.
        """
        from captioning.models.factory import build_caption_model

        tokenizer = CaptionTokenizer.load(
            directory=tokenizer_dir,
            vocab_size=config.model.vocabulary_size,
            max_length=config.model.max_length,
        )
        model = build_caption_model(config, vocab_size=tokenizer.vocabulary_size)
        # Build the model once before loading weights — Keras requires a
        # forward pass before ``load_weights`` knows variable shapes.
        cls._dummy_pass(model, config)
        model.load_weights(str(weights_path))

        log.info("predictor_loaded", weights=str(weights_path))
        return cls(model=model, tokenizer=tokenizer, config=config)

    def warmup(self) -> None:
        """Run one dummy inference so the first real request is fast."""
        import tensorflow as tf

        dummy = tf.zeros((299, 299, 3), dtype=tf.float32)
        _ = generate_caption_greedy(self.model, self.tokenizer, dummy, self.config.model.max_length)
        log.info("predictor_warmed_up")

    def predict_tensor(self, image_tensor) -> str:
        """Generate a caption from an already-preprocessed image tensor."""
        return generate_caption_greedy(
            self.model,
            self.tokenizer,
            image_tensor,
            self.config.model.max_length,
        )

    def predict_path(self, image_path: str | Path) -> str:
        """Generate a caption from an image on disk."""
        tensor = load_image_from_path(str(image_path))
        return self.predict_tensor(tensor)

    # ------------------------------------------------------------- internal --

    @staticmethod
    def _dummy_pass(model, config: AppConfig) -> None:
        """Force-build the model so ``load_weights`` knows variable shapes."""
        import tensorflow as tf

        dummy_img = tf.zeros((1, 299, 299, 3), dtype=tf.float32)
        dummy_caps = tf.zeros((1, config.model.max_length), dtype=tf.int64)
        # Calls train_step's underlying ops without doing a gradient step:
        img_embed = model.cnn_model(dummy_img)
        encoded = model.encoder(img_embed, training=False)
        _ = model.decoder(
            dummy_caps[:, :-1],
            encoded,
            training=False,
            mask=tf.cast(dummy_caps[:, 1:] != 0, tf.int32),
        )
