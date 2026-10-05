"""Pretrained Hugging Face captioning baselines as a :class:`Captioner`.

``torch`` and ``transformers`` (the optional ``[hf]`` extra) are imported only
inside :meth:`HFCaptioner.load`, through ``importlib``, so importing this
module never loads them (ADR-019).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from captioning.baselines.base import Captioner, CaptionerIdentity
from captioning.config.schema import BaselineDecodeConfig, ComparedModelConfig

HF_INSTALL_HINT = 'pip install -e ".[hf]"'


class MissingHFDependencyError(ImportError):
    """A Hugging Face baseline was used without the ``[hf]`` extra installed."""


@dataclass(frozen=True)
class _Loaded:
    torch: Any
    dtype: Any
    image_processor: Any
    tokenizer: Any
    model: Any


class HFCaptioner(Captioner):
    """One Hugging Face image-to-text model, pinned to an exact revision.

    BLIP-base, ViT-GPT2 and GIT-base-coco all load through
    ``AutoImageProcessor``, ``AutoTokenizer`` and ``AutoModelForVision2Seq``.
    Generation uses only the protocol's decode settings, passed explicitly to
    ``generate()``, with no text prompt (``EVAL_METHODOLOGY.md`` § 8.4).

    Construction is cheap and needs no ``[hf]`` packages. Weights load on
    :meth:`load`, or on the first :meth:`caption` call.
    """

    def __init__(
        self,
        model: ComparedModelConfig,
        decode: BaselineDecodeConfig,
        *,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            CaptionerIdentity(
                model_id=model.model_id,
                hub_repo=model.hub_repo,
                revision=model.revision,
                decode_settings=decode.model_dump(),
            )
        )
        self._model = model
        self._decode = decode
        self._device = device
        self._loaded: _Loaded | None = None

    @property
    def generate_kwargs(self) -> dict[str, Any]:
        """The exact keyword arguments passed to ``model.generate()``."""
        return {
            "num_beams": self._decode.num_beams,
            "do_sample": self._decode.do_sample,
            "max_new_tokens": self._decode.max_new_tokens,
            "repetition_penalty": self._decode.repetition_penalty,
            "no_repeat_ngram_size": self._decode.no_repeat_ngram_size,
        }

    def load(self) -> None:
        """Load the processor, tokenizer and model at the pinned revision.

        Raises:
            MissingHFDependencyError: If ``torch`` or ``transformers`` is not
                installed.
        """
        if self._loaded is not None:
            return
        torch, transformers = _import_hf()
        repo, revision = self._model.hub_repo, self._model.revision
        dtype = getattr(torch, self._decode.precision)
        image_processor = transformers.AutoImageProcessor.from_pretrained(repo, revision=revision)
        tokenizer = transformers.AutoTokenizer.from_pretrained(repo, revision=revision)
        model = transformers.AutoModelForVision2Seq.from_pretrained(
            repo, revision=revision, torch_dtype=dtype
        )
        model = model.to(self._device)
        model.eval()
        self._loaded = _Loaded(torch, dtype, image_processor, tokenizer, model)

    def _raw_captions(self, image_paths: list[Path]) -> list[str]:
        self.load()
        loaded = self._loaded
        assert loaded is not None
        images = [_open_rgb(path) for path in image_paths]
        pixel_values = loaded.image_processor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self._device, dtype=loaded.dtype)
        with loaded.torch.inference_mode():
            output_ids = loaded.model.generate(pixel_values=pixel_values, **self.generate_kwargs)
        return list(loaded.tokenizer.batch_decode(output_ids, skip_special_tokens=True))


def _import_hf() -> tuple[Any, Any]:
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        raise MissingHFDependencyError(
            "Hugging Face baselines need the optional [hf] extra (transformers, torch). "
            f"Install it with: {HF_INSTALL_HINT}"
        ) from exc
    return torch, transformers


def _open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")
