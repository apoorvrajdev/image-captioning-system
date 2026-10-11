"""Tests for bounded inference admission (TASK-027).

``PredictorService`` runs with a gated stand-in predictor: its
``predict_tensor`` blocks a real worker thread until the test opens the gate,
so requests genuinely overlap. Decoding runs the real TASK-026 header guard
and skips TensorFlow. Nothing sleeps: tests wait for the worker threads' own
signals, and ``fail_after`` only guards against a hang.
"""

from __future__ import annotations

import math
import threading
from typing import Any, cast

import anyio
import httpx
import pytest
from anyio import from_thread, to_thread

from app.api.routes import BUSY_DETAIL, BUSY_RETRY_AFTER_SECONDS
from app.services import predictor_service
from app.services.predictor_service import (
    MAX_CONCURRENT_INFERENCES,
    PredictorBusyError,
    PredictorService,
)
from app.tests.conftest import _build_app
from app.tests.image_bytes import encode, png_header
from app.utils.image import ImageTooLargeError, check_image_dimensions

IMAGE = encode("PNG")
HANG_GUARD_SECONDS = 10  # reached only if a test is broken
SLOTS = MAX_CONCURRENT_INFERENCES


class InferenceFailure(Exception):
    """A predictor failure that can't be mistaken for ``PredictorBusyError``."""


class GatedPredictor:
    """``predict_tensor`` holds its worker thread until ``open_gate()``."""

    decode_strategy = "greedy"

    def __init__(self) -> None:
        self.error: Exception | None = None
        self.running = 0
        self.peak = 0
        self.calls = 0
        self._gate = threading.Event()
        self._lock = threading.Lock()
        self._entered, self._entries = anyio.create_memory_object_stream[None](math.inf)

    def predict_tensor(self, tensor: object) -> str:
        with self._lock:
            self.calls += 1
            self.running += 1
            self.peak = max(self.peak, self.running)
        try:
            from_thread.run_sync(self._entered.send_nowait, None)
            if not self._gate.wait(HANG_GUARD_SECONDS):
                raise TimeoutError("the test never opened the gate")
            if self.error is not None:
                raise self.error
            return "a gated caption"
        finally:
            with self._lock:
                self.running -= 1

    def open_gate(self) -> None:
        self._gate.set()

    def close_gate(self) -> None:
        self._gate.clear()

    async def wait_running(self, count: int) -> None:
        """Return once ``count`` more calls have entered ``predict_tensor``."""
        for _ in range(count):
            await self._entries.receive()


@pytest.fixture(autouse=True)
def decoded(monkeypatch: pytest.MonkeyPatch) -> list[bytes]:
    """Replace the TF decode with the real header guard plus a placeholder tensor."""
    images: list[bytes] = []

    def decode(image_bytes: bytes) -> str:
        check_image_dimensions(image_bytes)
        images.append(image_bytes)
        return "tensor"

    monkeypatch.setattr(predictor_service, "bytes_to_tensor", decode)
    return images


def _service(predictor: GatedPredictor, slots: int = SLOTS) -> PredictorService:
    return PredictorService(
        predictor=cast(Any, predictor),
        model_version="test-v0",
        max_upload_bytes=64 * 1024,
        max_concurrent_inferences=slots,
    )


async def _saturate_then_release(
    service: PredictorService, predictor: GatedPredictor, slots: int = SLOTS
) -> list[str]:
    """Fill every slot with a blocked request, check one more is refused, then finish them."""
    captions: list[str] = []

    async def caption() -> None:
        text, _ = await service.caption_image_bytes(IMAGE)
        captions.append(text)

    with anyio.fail_after(HANG_GUARD_SECONDS):
        async with anyio.create_task_group() as tg:
            for _ in range(slots):
                tg.start_soon(caption)
            await predictor.wait_running(slots)
            assert service.inferences_in_flight == slots
            # Refused while every slot is still blocked, so it didn't wait for one.
            with pytest.raises(PredictorBusyError):
                await service.caption_image_bytes(IMAGE)
            predictor.open_gate()
    predictor.close_gate()
    assert service.inferences_in_flight == 0
    return captions


def test_rejects_a_limit_below_one() -> None:
    with pytest.raises(ValueError, match="max_concurrent_inferences"):
        _service(GatedPredictor(), slots=0)


def test_a_request_is_admitted_when_a_slot_is_free() -> None:
    async def main() -> None:
        predictor = GatedPredictor()
        predictor.open_gate()
        service = _service(predictor)
        with anyio.fail_after(HANG_GUARD_SECONDS):
            caption, latency_ms = await service.caption_image_bytes(IMAGE)
        assert caption == "a gated caption"
        assert latency_ms >= 0
        assert service.inferences_in_flight == 0

    anyio.run(main)


def test_a_request_over_capacity_is_refused_without_decoding(decoded: list[bytes]) -> None:
    async def main() -> None:
        predictor = GatedPredictor()
        service = _service(predictor)
        captions = await _saturate_then_release(service, predictor)
        assert captions == ["a gated caption"] * SLOTS
        assert len(decoded) == SLOTS  # the refused request decoded nothing
        assert predictor.calls == SLOTS
        assert predictor.peak == SLOTS

    anyio.run(main)


def test_slots_are_released_after_successful_inference() -> None:
    async def main() -> None:
        predictor = GatedPredictor()
        service = _service(predictor)
        await _saturate_then_release(service, predictor)
        # Every slot came back: a second full round is admitted the same way.
        await _saturate_then_release(service, predictor)
        assert predictor.calls == 2 * SLOTS
        assert predictor.peak == SLOTS

    anyio.run(main)


def test_slot_is_released_when_inference_raises() -> None:
    async def main() -> None:
        predictor = GatedPredictor()
        predictor.error = InferenceFailure("model failed")
        predictor.open_gate()
        service = _service(predictor)
        with anyio.fail_after(HANG_GUARD_SECONDS), pytest.raises(InferenceFailure):
            await service.caption_image_bytes(IMAGE)
        assert service.inferences_in_flight == 0

        predictor.error = None
        predictor.close_gate()
        await _saturate_then_release(service, predictor)

    anyio.run(main)


def test_slot_is_released_when_decoding_fails() -> None:
    async def main() -> None:
        predictor = GatedPredictor()
        service = _service(predictor)
        with anyio.fail_after(HANG_GUARD_SECONDS), pytest.raises(ImageTooLargeError):
            await service.caption_image_bytes(png_header(10_000, 5_000))
        assert service.inferences_in_flight == 0
        assert predictor.calls == 0

        await _saturate_then_release(service, predictor)

    anyio.run(main)


def test_a_cancelled_request_keeps_its_slot_until_its_thread_finishes() -> None:
    async def main() -> None:
        predictor = GatedPredictor()
        service = _service(predictor, slots=1)
        outcome: list[str] = []
        scope = anyio.CancelScope()

        async def request() -> None:
            with scope:
                await service.caption_image_bytes(IMAGE)
            outcome.append("ended")

        with anyio.fail_after(HANG_GUARD_SECONDS):
            async with anyio.create_task_group() as tg:
                tg.start_soon(request)
                await predictor.wait_running(1)
                scope.cancel()
                await anyio.wait_all_tasks_blocked()
                # Cancelled, but its thread is still running, so the slot stays taken
                # and no second inference can start beside it.
                assert outcome == []
                assert predictor.running == 1
                assert service.inferences_in_flight == 1
                with pytest.raises(PredictorBusyError):
                    await service.caption_image_bytes(IMAGE)
                predictor.open_gate()

        # It ended only once its thread had finished. The cancellation can't stop
        # that thread, and here it never surfaced: no checkpoint followed the wait.
        assert outcome == ["ended"]
        assert service.inferences_in_flight == 0
        assert predictor.running == 0
        assert predictor.peak == 1
        caption, _ = await service.caption_image_bytes(IMAGE)
        assert caption == "a gated caption"

    anyio.run(main)


def _upload(payload: bytes, content_type: str = "image/png") -> dict[str, Any]:
    return {"image": ("a.png", payload, content_type)}


def test_overload_is_a_fast_503_and_healthz_needs_no_worker_thread() -> None:
    async def main() -> None:
        predictor = GatedPredictor()
        service = _service(predictor)
        # Inference will hold every worker thread anyio has: a route that needed
        # one would hang here until the hang guard fails the test.
        to_thread.current_default_thread_limiter().total_tokens = SLOTS
        app = _build_app(cast(Any, service))
        # httpx types its ASGI app more narrowly than Starlette's own signature.
        transport = httpx.ASGITransport(app=cast(Any, app))
        responses: list[httpx.Response] = []

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:

            async def post() -> None:
                responses.append(await client.post("/v1/captions", files=_upload(IMAGE)))

            with anyio.fail_after(HANG_GUARD_SECONDS):
                async with anyio.create_task_group() as tg:
                    for _ in range(SLOTS):
                        tg.start_soon(post)
                    await predictor.wait_running(SLOTS)
                    assert to_thread.current_default_thread_limiter().available_tokens == 0

                    health = await client.get("/healthz")
                    assert health.status_code == 200
                    assert health.json()["status"] == "ok"
                    assert health.json()["model_loaded"] is True

                    busy = await client.post("/v1/captions", files=_upload(IMAGE))
                    assert busy.status_code == 503
                    assert busy.json()["detail"] == BUSY_DETAIL
                    assert busy.headers["retry-after"] == str(BUSY_RETRY_AFTER_SECONDS)
                    assert busy.headers["x-request-id"]

                    # Upload validation still runs before admission.
                    wrong_type = await client.post(
                        "/v1/captions", files=_upload(b"hello", "text/plain")
                    )
                    assert wrong_type.status_code == 415

                    predictor.open_gate()

        assert [response.status_code for response in responses] == [200] * SLOTS
        assert all(r.json()["caption"] == "a gated caption" for r in responses)
        assert predictor.peak == SLOTS

    anyio.run(main)
