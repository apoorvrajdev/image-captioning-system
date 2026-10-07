"""Time captioning latency through the shared Phase 3 :class:`Captioner` interface.

Implements the timing rules of ``docs/EVAL_METHODOLOGY.md`` § 9:

- One sample is the wall-clock time of one ``Captioner.caption()`` call on one
  batch of image paths, read from a monotonic clock. The call covers image
  reading and decoding, preprocessing, generation, token decoding and caption
  normalisation.
- Loading the model is timed once, separately, and is never part of a sample.
- For each batch size, the inputs are split into consecutive full batches in
  input order. Warmup passes caption every batch untimed, then each measured
  pass times every call.
- Nothing is filtered: every measured call is a sample, and any failed call
  ends the benchmark.

Nothing here imports TensorFlow, ``torch`` or ``transformers``.
"""

from __future__ import annotations

import importlib.metadata
import platform
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from captioning.baselines import Captioner

Clock = Callable[[], float]

LATENCY_PROTOCOL = "docs/EVAL_METHODOLOGY.md § 9"
CLOCK_NAME = "time.perf_counter"
# Distributions whose installed versions every latency run records.
RUNTIME_PACKAGES = ("tensorflow", "tensorflow-cpu", "torch", "transformers")


class LatencyBenchmarkError(RuntimeError):
    """A timed call broke the benchmark's rules, so no result is produced."""


@dataclass(frozen=True)
class LatencySettings:
    """What one latency run measures.

    Attributes:
        num_images: How many images, from the start of the slice, form the inputs.
        batch_sizes: Batch sizes to time, unique and ascending. Each must divide
            ``num_images``, so every call captions a full batch.
        warmup_passes: Untimed passes over the inputs before measuring, per
            batch size. At least one.
        measured_passes: Timed passes over the inputs, per batch size.
    """

    num_images: int
    batch_sizes: tuple[int, ...]
    warmup_passes: int
    measured_passes: int

    def __post_init__(self) -> None:
        if self.num_images < 1:
            raise ValueError(f"num_images must be at least 1, got {self.num_images}")
        if not self.batch_sizes:
            raise ValueError("at least one batch size is needed")
        if any(size < 1 for size in self.batch_sizes):
            raise ValueError(f"batch sizes must be at least 1, got {list(self.batch_sizes)}")
        if list(self.batch_sizes) != sorted(set(self.batch_sizes)):
            raise ValueError(
                f"batch sizes must be unique and ascending, got {list(self.batch_sizes)}"
            )
        uneven = [size for size in self.batch_sizes if self.num_images % size]
        if uneven:
            raise ValueError(
                f"num_images {self.num_images} is not a multiple of batch size(s) {uneven}; "
                "every timed call must caption a full batch"
            )
        if self.warmup_passes < 1:
            raise ValueError(
                f"at least one warmup pass is needed, got {self.warmup_passes}; "
                "the first call must never be timed"
            )
        if self.measured_passes < 1:
            raise ValueError(f"measured_passes must be at least 1, got {self.measured_passes}")

    def to_dict(self) -> dict[str, object]:
        return {
            "num_images": self.num_images,
            "batch_sizes": list(self.batch_sizes),
            "warmup_passes": self.warmup_passes,
            "measured_passes": self.measured_passes,
        }


@dataclass(frozen=True)
class BatchLatency:
    """Every timed call for one batch size, in measurement order."""

    batch_size: int
    calls_per_pass: int
    samples: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "calls_per_pass": self.calls_per_pass,
            "samples_seconds": list(self.samples),
            "summary_seconds": summarize(self.samples),
        }


def summarize(samples: Sequence[float]) -> dict[str, float | int]:
    """Return the § 9 statistics of ``samples``: count, mean, median, min and max."""
    if not samples:
        raise ValueError("no samples to summarise")
    return {
        "count": len(samples),
        "mean": statistics.fmean(samples),
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
    }


def time_load(
    factory: Callable[[], Captioner], *, clock: Clock = time.perf_counter
) -> tuple[Captioner, float]:
    """Build a captioner and load its weights, timing both steps together.

    The CNN + Transformer loads its checkpoint when it is constructed, and the
    Hugging Face adapters in ``load()``, so the load time covers both. It also
    includes any framework import or Hub download those steps trigger.
    """
    start = clock()
    captioner = factory()
    captioner.load()
    return captioner, _elapsed(start, clock())


def measure_latency(
    captioner: Captioner,
    image_paths: Sequence[str | Path],
    settings: LatencySettings,
    *,
    clock: Clock = time.perf_counter,
) -> list[BatchLatency]:
    """Time ``captioner`` on ``image_paths`` at every batch size in ``settings``.

    Batch sizes run in ascending order. For each one, the warmup passes run
    first and are never timed; each measured pass then times every batch, in
    input order. Only the ``caption()`` call sits between the two clock reads.

    Raises:
        ValueError: If ``image_paths`` doesn't hold exactly ``settings.num_images``
            paths.
        LatencyBenchmarkError: If a call returns the wrong number of captions,
            or the clock goes backwards.
        Exception: Whatever a failed ``caption()`` call raises, unchanged.
    """
    paths = [Path(p) for p in image_paths]
    if len(paths) != settings.num_images:
        raise ValueError(f"expected {settings.num_images} images, got {len(paths)}")

    results: list[BatchLatency] = []
    for batch_size in settings.batch_sizes:
        batches = [paths[i : i + batch_size] for i in range(0, len(paths), batch_size)]
        for _ in range(settings.warmup_passes):
            for batch in batches:
                _check_captions(captioner, batch, captioner.caption(batch))
        samples: list[float] = []
        for _ in range(settings.measured_passes):
            for batch in batches:
                start = clock()
                captions = captioner.caption(batch)
                end = clock()
                _check_captions(captioner, batch, captions)
                samples.append(_elapsed(start, end))
        results.append(BatchLatency(batch_size, len(batches), tuple(samples)))
    return results


def runtime_info() -> dict[str, object]:
    """Python, platform and installed framework versions (``None`` if not installed)."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: _installed_version(name) for name in RUNTIME_PACKAGES},
    }


def _installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _check_captions(captioner: Captioner, batch: list[Path], captions: Sequence[str]) -> None:
    if len(captions) != len(batch):
        raise LatencyBenchmarkError(
            f"{captioner.identity.model_id}: {len(captions)} captions for {len(batch)} images"
        )


def _elapsed(start: float, end: float) -> float:
    if end < start:
        raise LatencyBenchmarkError(f"the clock went backwards ({start} -> {end})")
    return end - start
