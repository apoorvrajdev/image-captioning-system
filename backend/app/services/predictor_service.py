"""Service layer wrapping the ML ``CaptionPredictor``.

Why this exists between the route and the predictor:
    * **Off-loop execution** — TensorFlow inference is sync and CPU-bound.
      Running it inline blocks the event loop, so requests queue up
      sequentially and event-loop-bound work (CORS, metrics, /healthz)
      stalls. We push the call to a worker thread via ``anyio.to_thread``.
    * **Stable seam for testing** — routes depend on this class, not on
      the concrete predictor. Tests can substitute a stub service that
      returns canned captions without loading TensorFlow.
    * **Future extension point** — Phase 4 will add a request batcher and
      per-model registry behind the same ``caption_image_bytes`` API.

This class never re-implements inference; it delegates entirely to the
existing ``CaptionPredictor`` abstraction.
"""

from __future__ import annotations

import time

from anyio import to_thread

from app.utils.image import bytes_to_tensor
from captioning.inference import CaptionPredictor
from captioning.utils import get_logger

log = get_logger(__name__)


class PredictorService:
    """Holds the singleton predictor and exposes async inference."""

    def __init__(
        self,
        *,
        predictor: CaptionPredictor,
        model_version: str,
        max_upload_bytes: int,
    ) -> None:
        """Args:
        predictor: A ready ``CaptionPredictor`` (weights already loaded).
        model_version: Semver string surfaced in responses & health.
        max_upload_bytes: Hard cap enforced at the route layer.
        """
        self._predictor = predictor
        self._model_version = model_version
        self._max_upload_bytes = max_upload_bytes

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def decode_strategy(self) -> str:
        return self._predictor.decode_strategy

    @property
    def max_upload_bytes(self) -> int:
        return self._max_upload_bytes

    async def caption_image_bytes(self, image_bytes: bytes) -> tuple[str, float]:
        """Decode bytes, run inference, and return (caption, latency_ms).

        Both the decode and the predict are offloaded to a worker thread so
        the event loop stays responsive. Latency is measured around the
        predict call only — decode timing belongs to a separate span if we
        ever need it.
        """
        tensor = await to_thread.run_sync(bytes_to_tensor, image_bytes)

        start = time.perf_counter()
        caption: str = await to_thread.run_sync(self._predictor.predict_tensor, tensor)
        latency_ms = (time.perf_counter() - start) * 1000

        log.info(
            "inference_completed",
            model_version=self._model_version,
            decode_strategy=self.decode_strategy,
            latency_ms=round(latency_ms, 2),
        )
        return caption, latency_ms
