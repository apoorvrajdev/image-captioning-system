"""The project's CNN + Transformer as a :class:`Captioner`."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from captioning.baselines.base import Captioner, CaptionerIdentity
from captioning.config.schema import AppConfig, ComparedModelConfig

if TYPE_CHECKING:
    from captioning.inference.predictor import DecodeStrategy


class _ModelSettings(Protocol):
    max_length: int


class _PredictorConfig(Protocol):
    @property
    def model(self) -> _ModelSettings: ...


class PathPredictor(Protocol):
    """The part of ``CaptionPredictor`` the adapter relies on."""

    decode_strategy: str
    beam_width: int
    length_penalty: float
    repetition_penalty: float
    no_repeat_ngram_size: int

    @property
    def config(self) -> _PredictorConfig: ...

    def predict_path(self, image_path: str | Path) -> str: ...


class CNNCaptioner(Captioner):
    """Wraps a loaded ``CaptionPredictor`` without changing how it decodes.

    Each image goes through ``predictor.predict_path``, one at a time, exactly
    as ``scripts/evaluate.py`` does. The recorded decode settings follow
    ``run_meta.json``: beam-only parameters are ``None`` for greedy decoding.
    """

    def __init__(self, predictor: PathPredictor, model: ComparedModelConfig) -> None:
        beam = predictor.decode_strategy == "beam"
        super().__init__(
            CaptionerIdentity(
                model_id=model.model_id,
                hub_repo=model.hub_repo,
                revision=model.revision,
                decode_settings={
                    "decode_strategy": predictor.decode_strategy,
                    "beam_width": predictor.beam_width if beam else None,
                    "length_penalty": predictor.length_penalty if beam else None,
                    "repetition_penalty": predictor.repetition_penalty,
                    "no_repeat_ngram_size": predictor.no_repeat_ngram_size,
                    "max_length": predictor.config.model.max_length,
                },
            )
        )
        self._predictor = predictor

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
    ) -> CNNCaptioner:
        """Load the checkpoint with ``CaptionPredictor.from_artifacts``.

        Decoding arguments are passed through unchanged, so unset ones fall
        back to ``config.serve`` just as they do for ``scripts/evaluate.py``.
        The identity comes from ``config.compare.cnn``. TensorFlow is imported
        here, not when this module is imported.
        """
        from captioning.inference.predictor import CaptionPredictor

        predictor: Any = CaptionPredictor.from_artifacts(
            weights_path,
            tokenizer_dir,
            config,
            decode_strategy=decode_strategy,
            beam_width=beam_width,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        return cls(predictor, config.compare.cnn)

    def _raw_captions(self, image_paths: list[Path]) -> list[str]:
        return [self._predictor.predict_path(path) for path in image_paths]
