"""The captioner interface shared by every Phase 3 model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from captioning.evaluation.tokenization import strip_sentinels
from captioning.preprocessing.caption import preprocess_caption


@dataclass(frozen=True)
class CaptionerIdentity:
    """What a run records about the model behind its captions.

    Attributes:
        model_id: The ``run_meta.json`` model id (``EVAL_METHODOLOGY.md`` § 8.1).
        hub_repo: Hugging Face Hub repository the weights come from, if any.
        revision: Pinned commit SHA of ``hub_repo``, if any.
        decode_settings: The decoding parameters actually used. Read-only.
    """

    model_id: str
    hub_repo: str | None
    revision: str | None
    decode_settings: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "decode_settings", MappingProxyType(dict(self.decode_settings)))


class Captioner(ABC):
    """Captions a batch of images and normalises the result.

    Subclasses produce raw captions; :meth:`caption` normalises every one of
    them through the existing training path (``preprocess_caption``, then
    ``strip_sentinels``), so all compared models are scored on the same
    footing (``EVAL_METHODOLOGY.md`` § 8.3).
    """

    def __init__(self, identity: CaptionerIdentity) -> None:
        self.identity = identity

    def load(self) -> None:  # noqa: B027 - optional hook, a no-op by default
        """Load model weights if they aren't loaded yet. Safe to call repeatedly."""

    def caption(self, image_paths: Sequence[str | Path]) -> list[str]:
        """Return one normalised caption per image, in input order."""
        paths = [Path(p) for p in image_paths]
        if not paths:
            return []
        raw = self._raw_captions(paths)
        if len(raw) != len(paths):
            raise RuntimeError(
                f"{self.identity.model_id}: {len(raw)} captions for {len(paths)} images"
            )
        return [strip_sentinels(preprocess_caption(text)) for text in raw]

    @abstractmethod
    def _raw_captions(self, image_paths: list[Path]) -> list[str]:
        """Return the model's captions for ``image_paths``, before normalisation."""
