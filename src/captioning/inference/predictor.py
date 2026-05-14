"""``CaptionPredictor`` — stateful, FastAPI-friendly inference singleton.

Why a class around the existing functions:
    * The FastAPI lifespan loads weights once at boot and reuses the same
      model across every request. A predictor object is the natural home for
      "loaded model + loaded tokenizer + decoded config".
    * Tests can construct one with stub objects without monkey-patching globals.
    * Multiple decode strategies (greedy, beam) live behind the same
      ``predict_tensor`` / ``predict_path`` API — callers do not need to know
      which one is active.

Construction is *not* the same as readiness: ``CaptionPredictor.warmup()``
runs one inference on a dummy tensor so the first real request doesn't pay
TF's lazy graph-build cost (typically 2-5 seconds).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from captioning.config.schema import AppConfig
from captioning.inference.beam import generate_caption_beam
from captioning.inference.greedy import generate_caption_greedy
from captioning.inference.image_loader import load_image_from_path
from captioning.preprocessing.tokenizer import CaptionTokenizer
from captioning.utils.logging import get_logger

log = get_logger(__name__)

DecodeStrategy = Literal["greedy", "beam"]


class CaptionPredictor:
    """Thin wrapper exposing ``predict_path`` / ``predict_tensor`` / ``warmup``."""

    def __init__(
        self,
        model,
        tokenizer: CaptionTokenizer,
        config: AppConfig,
        *,
        decode_strategy: DecodeStrategy = "greedy",
        beam_width: int = 3,
        length_penalty: float = 1.0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
    ) -> None:
        """Args:
        model: Loaded ``ImageCaptioningModel``. Caller is responsible for
            having called ``model.load_weights(...)`` already.
        tokenizer: Fitted ``CaptionTokenizer``.
        config: Validated ``AppConfig`` — ``model.max_length`` is consumed.
        decode_strategy: ``"greedy"`` (argmax per step, byte-for-byte parity
            with the IEEE notebook) or ``"beam"`` (beam search with length
            and repetition controls).
        beam_width: Beam width when ``decode_strategy == "beam"``. Ignored
            for greedy.
        length_penalty: GNMT length penalty; ``0.0`` disables, ``0.6-1.0`` is
            the common range.
        repetition_penalty: HF-style multiplicative penalty on already-seen
            tokens; ``1.0`` disables.
        no_repeat_ngram_size: If > 0, blocks any token that would repeat an
            n-gram already in the partial caption.
        """
        if decode_strategy not in {"greedy", "beam"}:
            raise ValueError(f"decode_strategy must be 'greedy' or 'beam', got {decode_strategy!r}")
        if beam_width < 1:
            raise ValueError(f"beam_width must be >= 1, got {beam_width}")
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.decode_strategy: DecodeStrategy = decode_strategy
        self.beam_width = beam_width
        self.length_penalty = length_penalty
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size

    @classmethod
    def from_artifacts(
        cls,
        weights_path: str | Path,
        tokenizer_dir: str | Path,
        config: AppConfig,
        *,
        decode_strategy: DecodeStrategy | None = None,
        beam_width: int | None = None,
        length_penalty: float | None = None,
        repetition_penalty: float | None = None,
        no_repeat_ngram_size: int | None = None,
    ) -> CaptionPredictor:
        """Load weights and tokenizer from disk and return a ready predictor.

        Decoding knobs fall back to :class:`ServeConfig` defaults when not
        passed explicitly — keeping CLI flags overridable while still letting
        deploy-time YAML drive the production behaviour.
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

        resolved_strategy: DecodeStrategy = (
            decode_strategy or config.serve.decode_strategy  # type: ignore[assignment]
        )
        log.info(
            "predictor_loaded",
            weights=str(weights_path),
            decode_strategy=resolved_strategy,
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            config=config,
            decode_strategy=resolved_strategy,
            beam_width=beam_width if beam_width is not None else config.serve.beam_width,
            length_penalty=(
                length_penalty if length_penalty is not None else config.serve.length_penalty
            ),
            repetition_penalty=(
                repetition_penalty
                if repetition_penalty is not None
                else config.serve.repetition_penalty
            ),
            no_repeat_ngram_size=(
                no_repeat_ngram_size
                if no_repeat_ngram_size is not None
                else config.serve.no_repeat_ngram_size
            ),
        )

    def warmup(self) -> None:
        """Run one dummy inference so the first real request is fast."""
        import tensorflow as tf

        dummy = tf.zeros((299, 299, 3), dtype=tf.float32)
        _ = self.predict_tensor(dummy)
        log.info("predictor_warmed_up", decode_strategy=self.decode_strategy)

    def predict_tensor(self, image_tensor) -> str:
        """Generate a caption from an already-preprocessed image tensor."""
        if self.decode_strategy == "beam":
            return generate_caption_beam(
                self.model,
                self.tokenizer,
                image_tensor,
                self.config.model.max_length,
                beam_width=self.beam_width,
                length_penalty=self.length_penalty,
                repetition_penalty=self.repetition_penalty,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
            )
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
        """Force-build the model so ``load_weights`` knows variable shapes.

        ``ImageCaptioningModel`` has no top-level ``call()`` — it overrides
        ``train_step``/``test_step`` instead. Keras therefore won't mark the
        parent ``Model`` as ``built`` even after every sublayer has its
        variables created, and the HDF5 ``load_weights`` path refuses to
        proceed against an unbuilt subclassed model. We work around this by
        (a) calling each sublayer once so its variables are real (shape-
        matched to the saved checkpoint) and (b) flipping ``model.built``
        so the loader walks the sublayer scopes inside the file. The actual
        weights loaded are still those from the checkpoint — this is purely
        a Keras bookkeeping flag.
        """
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
        # Augmentation pipeline is tracked as a sublayer of the parent Model
        # even though inference never invokes it; building it once keeps the
        # variable tree identical to what `model.fit` produced when Phase 1
        # weights were saved.
        if getattr(model, "image_aug", None) is not None:
            _ = model.image_aug(dummy_img, training=False)
        # Sublayers are now built; mark the parent built so HDF5 load_weights
        # accepts the file. Safe because every variable that the checkpoint
        # references is already materialised on a tracked sublayer.
        model.built = True
