"""Service layer wrapping the ML ``CaptionPredictor``.

Why this exists between the route and the predictor:
    * **Off-loop execution** — TensorFlow inference is sync and CPU-bound.
      Running it inline blocks the event loop, so requests queue up
      sequentially and event-loop-bound work (CORS, metrics, /healthz)
      stalls. We push the call to a worker thread via ``anyio.to_thread``.
    * **Bounded admission** — at most ``max_concurrent_inferences`` requests
      decode or predict at once. The next one is refused straight away
      (``PredictorBusyError``, a 503 at the route) instead of waiting.
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
from functools import partial

from anyio import to_thread

from app.utils.image import bytes_to_tensor
from captioning.inference import CaptionPredictor
from captioning.inference.predictor import DECODE_STRATEGIES
from captioning.utils import get_logger

log = get_logger(__name__)

__all__ = [
    "DECODE_STRATEGIES",
    "MAX_CONCURRENT_INFERENCES",
    "PredictorBusyError",
    "PredictorService",
]

# Requests allowed to decode or predict at once. The Space runs on HF's
# cpu-basic tier (2 vCPU), and TensorFlow already spreads one inference across
# both cores, so more parallel inferences would compete for CPU rather than add
# throughput. Two also bounds the decode working set to about 1.2 GB by the
# TASK-026 estimate (~600 MB per image at the pixel limit). Chosen from the
# hardware, not benchmarked.
MAX_CONCURRENT_INFERENCES = 2


class PredictorBusyError(RuntimeError):
    """Raised when every inference slot is in use. The request isn't queued."""


class PredictorService:
    """Holds the singleton predictor and exposes async inference."""

    def __init__(
        self,
        *,
        predictor: CaptionPredictor,
        model_version: str,
        max_upload_bytes: int,
        max_concurrent_inferences: int = MAX_CONCURRENT_INFERENCES,
    ) -> None:
        """Args:
        predictor: A ready ``CaptionPredictor`` (weights already loaded).
        model_version: Semver string surfaced in responses & health.
        max_upload_bytes: Hard cap enforced at the route layer.
        max_concurrent_inferences: Inference slots; at least 1.
        """
        if max_concurrent_inferences < 1:
            raise ValueError(
                f"max_concurrent_inferences must be >= 1, got {max_concurrent_inferences}"
            )
        self._predictor = predictor
        self._model_version = model_version
        self._max_upload_bytes = max_upload_bytes
        self._max_concurrent = max_concurrent_inferences
        self._in_flight = 0

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def decode_strategy(self) -> str:
        return self._predictor.decode_strategy

    @property
    def max_upload_bytes(self) -> int:
        return self._max_upload_bytes

    @property
    def inferences_in_flight(self) -> int:
        """Requests holding a slot, including cancelled ones whose thread still runs."""
        return self._in_flight

    async def caption_image_bytes(
        self, image_bytes: bytes, *, decode_strategy: str | None = None
    ) -> tuple[str, float]:
        """Decode bytes, run inference, and return (caption, latency_ms).

        ``decode_strategy`` applies to this call only; ``None`` means the
        predictor's default. The route validates the value first.

        Both the decode and the predict are offloaded to a worker thread so
        the event loop stays responsive. Latency is measured around the
        predict call only — decode timing belongs to a separate span if we
        ever need it.

        Admission: the slot check and the increment run on the event loop
        with no ``await`` between them, so two requests can't both take the
        last slot. ``finally`` gives the slot back on every exit path.

        Cancellation: ``to_thread.run_sync`` keeps its default
        ``abandon_on_cancel=False``, so a cancelled caller (a client
        disconnect, for instance) still waits for the worker thread it
        started, and its slot is freed only when that thread has finished.
        Cancelling doesn't stop TensorFlow work that is already running.

        Raises:
            PredictorBusyError: If every slot is in use. Nothing is decoded.
        """
        if self._in_flight >= self._max_concurrent:
            raise PredictorBusyError("All inference slots are in use.")
        self._in_flight += 1
        try:
            tensor = await to_thread.run_sync(bytes_to_tensor, image_bytes)

            start = time.perf_counter()
            caption: str = await to_thread.run_sync(
                partial(self._predictor.predict_tensor, tensor, decode_strategy=decode_strategy)
            )
            latency_ms = (time.perf_counter() - start) * 1000
        finally:
            self._in_flight -= 1

        log.info(
            "inference_completed",
            model_version=self._model_version,
            decode_strategy=decode_strategy or self.decode_strategy,
            latency_ms=round(latency_ms, 2),
        )
        return caption, latency_ms
