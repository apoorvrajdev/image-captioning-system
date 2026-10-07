"""Tests for the Phase 3 latency benchmark (TASK-015).

Offline: captioners are fakes (the HF ones keep the real :class:`HFCaptioner`
identity), and time comes from a scripted clock that only moves while a fake
model "works". Every sample is therefore exact, so the tests can check which
calls were timed, the statistics and the written file without real inference.
"""

from __future__ import annotations

import importlib.metadata
import json
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import scripts.benchmark_latency as benchmark_latency
from click.testing import CliRunner, Result

from captioning.baselines import Captioner, CaptionerIdentity, CNNCaptioner, HFCaptioner
from captioning.config import AppConfig, BaselineDecodeConfig, ComparedModelConfig
from captioning.evaluation import (
    LatencyBenchmarkError,
    LatencySettings,
    load_eval_slice,
    measure_latency,
    runtime_info,
    slice_fingerprint,
    summarize,
    time_load,
)
from captioning.evaluation import latency as latency_module

REPO_ROOT = Path(__file__).resolve().parents[2]
# Five rows and eight references; the benchmark uses the first four images.
FIXTURE_REFERENCE_COUNTS = [2, 1, 3, 1, 1]
QUALITY_FILES = {
    "metrics.json",
    "predictions.jsonl",
    "diagnostics.jsonl",
    "run_meta.json",
    "report.md",
    "comparison_meta.json",
}
WARMUP = 100.0  # every warmup call takes this long, so a leaked warmup sample is obvious
LOAD = 7.5


class ScriptedClock:
    """A monotonic clock that only moves when a fake model advances it."""

    def __init__(self) -> None:
        self.now = 0.0
        self.reads = 0

    def __call__(self) -> float:
        self.reads += 1
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TimedFakeCaptioner(Captioner):
    """Takes ``durations[i]`` seconds for its i-th call and logs every batch."""

    def __init__(
        self,
        clock: ScriptedClock,
        durations: Sequence[float],
        *,
        fail_on_call: int | None = None,
    ) -> None:
        super().__init__(CaptionerIdentity("fake-model", None, None, {"decode_strategy": "greedy"}))
        self.clock = clock
        self.durations = list(durations)
        self.fail_on_call = fail_on_call
        self.calls: list[list[str]] = []

    def _raw_captions(self, image_paths: list[Path]) -> list[str]:
        index = len(self.calls)
        self.calls.append([p.name for p in image_paths])
        if index == self.fail_on_call:
            raise RuntimeError("model crashed")
        self.clock.advance(self.durations[index])
        return [f"A photo of {p.stem}." for p in image_paths]


def _durations(warmup_calls: int, measured: Sequence[float]) -> list[float]:
    return [WARMUP] * warmup_calls + list(measured)


IMAGES = [f"img{i}.jpg" for i in range(4)]
# Batch sizes 1 and 2 over four images, one warmup pass and two measured passes.
SETTINGS = LatencySettings(num_images=4, batch_sizes=(1, 2), warmup_passes=1, measured_passes=2)
MEASURED_B1 = [0.5, 0.25, 0.75, 0.5, 0.25, 0.25, 2.0, 0.25]
MEASURED_B2 = [1.5, 1.0, 1.25, 1.0]
SCRIPT = _durations(4, MEASURED_B1) + _durations(2, MEASURED_B2)


# ----------------------------------------------------------------- statistics --


def test_summarize_matches_hand_computed_values() -> None:
    assert summarize([0.5, 0.25, 0.75, 0.5, 0.25, 0.25, 2.0, 0.25]) == {
        "count": 8,
        "mean": 0.59375,  # 4.75 / 8
        "median": 0.375,  # (0.25 + 0.5) / 2
        "min": 0.25,
        "max": 2.0,
    }
    assert summarize([3.0, 1.0, 2.0]) == {
        "count": 3,
        "mean": 2.0,
        "median": 2.0,
        "min": 1.0,
        "max": 3.0,
    }


def test_summarize_rejects_no_samples() -> None:
    with pytest.raises(ValueError, match="no samples"):
        summarize([])


# ------------------------------------------------------------------- settings --


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_images": 0}, "num_images must be at least 1"),
        ({"batch_sizes": ()}, "at least one batch size"),
        ({"batch_sizes": (0, 1)}, "at least 1"),
        ({"batch_sizes": (2, 1)}, "unique and ascending"),
        ({"batch_sizes": (1, 1)}, "unique and ascending"),
        ({"num_images": 6, "batch_sizes": (1, 4)}, r"multiple of batch size\(s\) \[4\]"),
        ({"warmup_passes": 0}, "at least one warmup pass"),
        ({"measured_passes": 0}, "measured_passes must be at least 1"),
    ],
)
def test_settings_reject_invalid_values(kwargs: dict[str, Any], message: str) -> None:
    values: dict[str, Any] = {
        "num_images": 4,
        "batch_sizes": (1, 2),
        "warmup_passes": 1,
        "measured_passes": 2,
    }
    with pytest.raises(ValueError, match=message):
        LatencySettings(**{**values, **kwargs})


# ---------------------------------------------------------------- measurement --


def test_warmup_is_excluded_and_every_measured_call_is_one_sample() -> None:
    clock = ScriptedClock()
    captioner = TimedFakeCaptioner(clock, SCRIPT)

    results = measure_latency(captioner, IMAGES, SETTINGS, clock=clock)

    assert [r.batch_size for r in results] == [1, 2]
    assert [r.calls_per_pass for r in results] == [4, 2]
    assert list(results[0].samples) == MEASURED_B1
    assert list(results[1].samples) == MEASURED_B2
    assert WARMUP not in results[0].samples + results[1].samples
    # The clock is read only around measured calls: two reads per sample.
    assert clock.reads == 2 * (len(MEASURED_B1) + len(MEASURED_B2))


def test_inputs_run_in_slice_order_with_warmup_first_for_each_batch_size() -> None:
    clock = ScriptedClock()
    captioner = TimedFakeCaptioner(clock, SCRIPT)

    measure_latency(captioner, IMAGES, SETTINGS, clock=clock)

    singles = [[name] for name in IMAGES]
    pairs = [IMAGES[0:2], IMAGES[2:4]]
    assert captioner.calls == singles * 3 + pairs * 3  # 1 warmup + 2 measured passes each


def test_batch_results_serialise_samples_and_statistics() -> None:
    clock = ScriptedClock()
    results = measure_latency(TimedFakeCaptioner(clock, SCRIPT), IMAGES, SETTINGS, clock=clock)

    assert results[1].to_dict() == {
        "batch_size": 2,
        "calls_per_pass": 2,
        "samples_seconds": MEASURED_B2,
        "summary_seconds": {
            "count": 4,
            "mean": 1.1875,
            "median": 1.125,
            "min": 1.0,
            "max": 1.5,
        },
    }


@pytest.mark.parametrize("failing_call", [0, 5])  # a warmup call, then a measured call
def test_a_failed_call_fails_the_benchmark(failing_call: int) -> None:
    clock = ScriptedClock()
    captioner = TimedFakeCaptioner(clock, SCRIPT, fail_on_call=failing_call)

    with pytest.raises(RuntimeError, match="model crashed"):
        measure_latency(captioner, IMAGES, SETTINGS, clock=clock)
    assert len(captioner.calls) == failing_call + 1  # nothing ran after the failure


def test_a_call_with_missing_captions_fails_the_benchmark() -> None:
    class DroppingCaptioner(TimedFakeCaptioner):
        def caption(self, image_paths: Sequence[str | Path]) -> list[str]:
            return super().caption(image_paths)[:-1]

    clock = ScriptedClock()
    with pytest.raises(LatencyBenchmarkError, match="0 captions for 1 images"):
        measure_latency(DroppingCaptioner(clock, SCRIPT), IMAGES, SETTINGS, clock=clock)


def test_a_clock_that_goes_backwards_fails_the_benchmark() -> None:
    clock = ScriptedClock()
    with pytest.raises(LatencyBenchmarkError, match="went backwards"):
        measure_latency(
            TimedFakeCaptioner(clock, [WARMUP] * 4 + [-1.0]), IMAGES, SETTINGS, clock=clock
        )


def test_the_input_count_must_match_the_settings() -> None:
    clock = ScriptedClock()
    with pytest.raises(ValueError, match="expected 4 images, got 3"):
        measure_latency(TimedFakeCaptioner(clock, SCRIPT), IMAGES[:3], SETTINGS, clock=clock)


def test_load_time_covers_construction_and_load_only() -> None:
    clock = ScriptedClock()
    loads: list[int] = []

    class SlowLoadingCaptioner(TimedFakeCaptioner):
        def load(self) -> None:
            loads.append(1)
            clock.advance(3.0)

    def factory() -> Captioner:
        clock.advance(2.0)  # the CNN loads its checkpoint while it is constructed
        return SlowLoadingCaptioner(clock, SCRIPT)

    captioner, seconds = time_load(factory, clock=clock)

    assert seconds == 5.0
    assert loads == [1]
    assert isinstance(captioner, SlowLoadingCaptioner)
    assert captioner.calls == []  # loading never captions


def test_runtime_info_records_missing_packages_as_none(monkeypatch: pytest.MonkeyPatch) -> None:
    installed = {"tensorflow-cpu": "2.15.0", "transformers": "4.41.2"}

    def version(name: str) -> str:
        if name not in installed:
            raise importlib.metadata.PackageNotFoundError(name)
        return installed[name]

    monkeypatch.setattr(latency_module.importlib.metadata, "version", version)

    info = runtime_info()

    assert set(info) == {"python", "platform", "packages"}
    assert info["packages"] == {
        "tensorflow": None,
        "tensorflow-cpu": "2.15.0",
        "torch": None,
        "transformers": "4.41.2",
    }


# ------------------------------------------------------------------------- CLI --


class FakeHFCaptioner(HFCaptioner):
    """Real HF identity and decode settings; scripted timing, no model."""

    def __init__(
        self,
        model: ComparedModelConfig,
        decode: BaselineDecodeConfig,
        *,
        device: str,
        clock: ScriptedClock,
        fail_on_call: int | None = None,
    ) -> None:
        super().__init__(model, decode, device=device)
        self.device = device
        self.fake = TimedFakeCaptioner(clock, SCRIPT, fail_on_call=fail_on_call)
        self.clock = clock
        self.loads = 0

    def load(self) -> None:
        self.loads += 1
        self.clock.advance(LOAD)

    def _raw_captions(self, image_paths: list[Path]) -> list[str]:
        return self.fake._raw_captions(image_paths)


@pytest.fixture
def workspace(tmp_path: Path) -> dict[str, Path]:
    rows = [
        {
            "image": f"/kaggle/coco2017/train2017/img{i}.jpg",
            "prediction": "x",
            "references": [f"[start] reference {j} of image {i} [end]" for j in range(n_refs)],
        }
        for i, n_refs in enumerate(FIXTURE_REFERENCE_COUNTS)
    ]
    slice_path = tmp_path / "slice" / "predictions.jsonl"
    slice_path.parent.mkdir()
    slice_path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    for i in range(len(rows)):
        (images_dir / f"img{i}.jpg").write_bytes(b"")  # existence only; fakes don't read
    cnn_dir = tmp_path / "cnn"
    cnn_dir.mkdir()
    (cnn_dir / "model.h5").write_bytes(b"")
    return {
        "slice": slice_path,
        "images": images_dir,
        "results": tmp_path / "results",
        "cnn_weights": cnn_dir / "model.h5",
        "cnn_dir": cnn_dir,
    }


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Replaces the clock and the model build; ``fail_on_call`` scripts a crash."""
    state = SimpleNamespace(clock=ScriptedClock(), built=[], fail_on_call=None)
    monkeypatch.setattr(benchmark_latency, "CLOCK", state.clock)

    def build(spec: Any, config: AppConfig, **kwargs: Any) -> Captioner:
        captioner = FakeHFCaptioner(
            spec.model,
            config.compare.baseline_decode,
            device=kwargs["device"],
            clock=state.clock,
            fail_on_call=state.fail_on_call,
        )
        state.built.append(captioner)
        return captioner

    monkeypatch.setattr(benchmark_latency, "build_captioner", build)
    return state


def _invoke(workspace: dict[str, Path], *args: str, model: str = "blip-base") -> Result:
    base = [
        "--config",
        str(REPO_ROOT / "configs" / "base.yaml"),
        "--slice",
        str(workspace["slice"]),
        "--images-dir",
        str(workspace["images"]),
        "--results-root",
        str(workspace["results"]),
        "--expected-images",
        "5",
        "--expected-references",
        "8",
        "--model",
        model,
        "--num-images",
        "4",
        "--batch-size",
        "2",
        "--batch-size",
        "1",
        "--warmup-passes",
        "1",
        "--measured-passes",
        "2",
        "--environment",
        "test host, no accelerator",
    ]
    if "--device" not in args:
        base += ["--device", "cpu"]
    return CliRunner().invoke(benchmark_latency.main, [*base, *args])


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_writes_one_latency_json_with_the_documented_shape(
    workspace: dict[str, Path], harness: SimpleNamespace
) -> None:
    result = _invoke(workspace)
    assert result.exit_code == 0, result.output

    run_dir = workspace["results"] / "phase3-latency-blip-base-greedy-cpu"
    # Latency only: no quality-evaluation file is written next to it.
    assert sorted(p.name for p in run_dir.iterdir()) == ["latency.json"]
    assert not QUALITY_FILES & {p.name for p in run_dir.iterdir()}
    text = (run_dir / "latency.json").read_text(encoding="utf-8")
    assert text.endswith("}\n")
    record = json.loads(text)

    assert list(record) == [
        "protocol",
        "model_id",
        "backend",
        "captioner",
        "hub_repo",
        "revision",
        "decode_strategy",
        "decode_settings",
        "device",
        "environment",
        "inputs",
        "settings",
        "timing",
        "load_seconds",
        "batches",
        "seed",
        "runtime",
    ]
    compare = AppConfig().compare
    blip = next(m for m in compare.baselines if m.model_id == "blip-base")
    assert record["protocol"] == "docs/EVAL_METHODOLOGY.md § 9"
    assert (record["model_id"], record["hub_repo"], record["revision"]) == (
        "blip-base",
        blip.hub_repo,
        blip.revision,
    )
    assert (record["backend"], record["captioner"]) == ("huggingface", "FakeHFCaptioner")
    assert record["decode_strategy"] == "greedy"
    assert record["decode_settings"] == compare.baseline_decode.model_dump()
    assert (record["device"], record["environment"]) == ("cpu", "test host, no accelerator")
    assert record["seed"] == 42

    eval_slice = load_eval_slice(workspace["slice"], workspace["images"])
    assert record["inputs"] == {
        "slice": {
            "source": workspace["slice"].as_posix(),
            "fingerprint_sha256": slice_fingerprint(eval_slice),
            "images": 5,
            "references": 8,
        },
        "images": IMAGES,
    }
    assert record["settings"] == {
        "num_images": 4,
        "batch_sizes": [1, 2],  # given as 2, 1: always run and stored ascending
        "warmup_passes": 1,
        "measured_passes": 2,
    }
    assert record["timing"]["clock"] == "time.perf_counter"
    assert set(record["timing"]) == {"clock", "sample", "load"}

    # Load time is its own number; warmup calls never become samples.
    assert record["load_seconds"] == LOAD
    assert [b["samples_seconds"] for b in record["batches"]] == [MEASURED_B1, MEASURED_B2]
    assert [b["calls_per_pass"] for b in record["batches"]] == [4, 2]
    assert record["batches"][0]["summary_seconds"] == summarize(MEASURED_B1)
    assert record["batches"][1]["summary_seconds"] == summarize(MEASURED_B2)

    runtime = record["runtime"]
    assert set(runtime) == {"python", "platform", "packages", "tensorflow_gpus"}
    assert set(runtime["packages"]) == {"tensorflow", "tensorflow-cpu", "torch", "transformers"}
    assert runtime["tensorflow_gpus"] is None  # only checked for the CNN

    (captioner,) = harness.built
    assert captioner.loads == 1
    assert captioner.device == "cpu"


def test_output_is_byte_identical_for_identical_timings(
    workspace: dict[str, Path], harness: SimpleNamespace
) -> None:
    first = _invoke(workspace, "--run-prefix", "a-")
    second = _invoke(workspace, "--run-prefix", "b-")
    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output

    a = workspace["results"] / "a-blip-base-greedy-cpu" / "latency.json"
    b = workspace["results"] / "b-blip-base-greedy-cpu" / "latency.json"
    assert a.read_bytes() == b.read_bytes()


def test_the_device_reaches_the_adapter_and_names_the_run(
    workspace: dict[str, Path], harness: SimpleNamespace
) -> None:
    result = _invoke(workspace, "--device", "cuda")
    assert result.exit_code == 0, result.output

    record = _read(workspace["results"] / "phase3-latency-blip-base-greedy-cuda" / "latency.json")
    assert record["device"] == "cuda"
    assert harness.built[0].device == "cuda"


def _assert_nothing_ran(workspace: dict[str, Path], harness: SimpleNamespace) -> None:
    assert harness.built == []
    assert not workspace["results"].exists()


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("--num-images", "3"), r"not a multiple of batch size\(s\) \[2\]"),
        (("--batch-size", "1"), "unique and ascending"),
        (("--warmup-passes", "0"), "at least one warmup pass"),
        (("--measured-passes", "0"), "measured_passes must be at least 1"),
        (("--num-images", "6"), "--num-images 6 exceeds the slice's 5 images"),
        (("--expected-images", "500"), "expected 500 images"),
        (("--environment", "  "), "--environment must name"),
        (("--model", "no-such-model"), "unknown model id"),
        (("--device", "tpu"), "Invalid value for '--device'"),
    ],
)
def test_invalid_settings_fail_before_any_model_loads(
    workspace: dict[str, Path], harness: SimpleNamespace, args: tuple[str, ...], message: str
) -> None:
    result = _invoke(workspace, *args)

    assert result.exit_code != 0
    assert re.search(message, result.output), result.output
    _assert_nothing_ran(workspace, harness)


def test_the_cnn_needs_its_checkpoint(workspace: dict[str, Path], harness: SimpleNamespace) -> None:
    result = _invoke(workspace, model="inceptionv3-transformer-stabilized")

    assert result.exit_code != 0
    assert "--cnn-weights and --cnn-tokenizer-dir" in result.output
    _assert_nothing_ran(workspace, harness)


def test_existing_run_directory_is_never_overwritten(
    workspace: dict[str, Path], harness: SimpleNamespace
) -> None:
    run_dir = workspace["results"] / "phase3-latency-blip-base-greedy-cpu"
    run_dir.mkdir(parents=True)
    (run_dir / "latency.json").write_text("{}", encoding="utf-8")

    result = _invoke(workspace)

    assert result.exit_code != 0
    assert "run directory already exists" in result.output
    assert harness.built == []
    assert (run_dir / "latency.json").read_text(encoding="utf-8") == "{}"


def test_only_the_benchmark_images_must_exist(
    workspace: dict[str, Path], harness: SimpleNamespace
) -> None:
    (workspace["images"] / "img4.jpg").unlink()  # not among the first four
    assert _invoke(workspace, "--run-prefix", "ok-").exit_code == 0

    (workspace["images"] / "img2.jpg").unlink()
    harness.built.clear()
    result = _invoke(workspace)

    assert result.exit_code != 0
    assert "1 of 4 benchmark images are missing" in result.output
    assert "img2.jpg" in result.output
    assert harness.built == []
    assert not (workspace["results"] / "phase3-latency-blip-base-greedy-cpu").exists()


@pytest.mark.parametrize("failing_call", [1, 6])  # a warmup call, then a measured call
def test_a_failed_call_writes_nothing(
    workspace: dict[str, Path], harness: SimpleNamespace, failing_call: int
) -> None:
    harness.fail_on_call = failing_call

    result = _invoke(workspace)

    assert result.exit_code != 0
    assert isinstance(result.exception, RuntimeError)
    assert not workspace["results"].exists()


# ----------------------------------------------------------- CNN device check --


class _FakeCNNPredictor:
    decode_strategy = "greedy"
    beam_width = 3
    length_penalty = 1.0
    repetition_penalty = 1.0
    no_repeat_ngram_size = 0

    def __init__(self, clock: ScriptedClock) -> None:
        self.config = SimpleNamespace(model=SimpleNamespace(max_length=40))
        self.clock = clock

    def predict_path(self, image_path: str | Path) -> str:
        self.clock.advance(0.25)
        return f"a caption for {Path(image_path).stem}"


@pytest.fixture
def cnn_harness(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[list[str]], SimpleNamespace]:
    def install(tensorflow_gpus: list[str]) -> SimpleNamespace:
        state = SimpleNamespace(clock=ScriptedClock(), built=[])
        monkeypatch.setattr(benchmark_latency, "CLOCK", state.clock)
        monkeypatch.setattr(benchmark_latency, "tensorflow_gpus", lambda: list(tensorflow_gpus))

        def build(spec: Any, config: AppConfig, **kwargs: Any) -> Captioner:
            assert kwargs["cnn_weights"] is not None
            captioner = CNNCaptioner(_FakeCNNPredictor(state.clock), config.compare.cnn)
            state.built.append(captioner)
            return captioner

        monkeypatch.setattr(benchmark_latency, "build_captioner", build)
        return state

    return install


def _invoke_cnn(workspace: dict[str, Path], device: str) -> Result:
    return _invoke(
        workspace,
        "--device",
        device,
        "--cnn-weights",
        str(workspace["cnn_weights"]),
        "--cnn-tokenizer-dir",
        str(workspace["cnn_dir"]),
        model="inceptionv3-transformer-stabilized",
    )


def test_the_cnn_runs_when_its_device_matches_tensorflow(
    workspace: dict[str, Path], cnn_harness: Callable[[list[str]], SimpleNamespace]
) -> None:
    cnn_harness([])

    result = _invoke_cnn(workspace, "cpu")
    assert result.exit_code == 0, result.output

    run_dir = workspace["results"] / "phase3-latency-inceptionv3-transformer-stabilized-greedy-cpu"
    record = _read(run_dir / "latency.json")
    cnn = AppConfig().compare.cnn
    assert (record["backend"], record["captioner"]) == ("cnn", "CNNCaptioner")
    assert (record["hub_repo"], record["revision"]) == (cnn.hub_repo, cnn.revision)
    assert record["decode_settings"]["decode_strategy"] == "greedy"
    assert record["runtime"]["tensorflow_gpus"] == []
    # The CNN captions a batch one image at a time: a batch of 2 takes two predictions.
    assert record["batches"][0]["samples_seconds"] == [0.25] * 8
    assert record["batches"][1]["samples_seconds"] == [0.5] * 4


@pytest.mark.parametrize(
    ("device", "gpus", "message"),
    [
        ("cpu", ["/physical_device:GPU:0"], "TensorFlow can see /physical_device:GPU:0"),
        ("cuda", [], "TensorFlow can see no GPU"),
    ],
)
def test_the_cnn_device_must_match_what_tensorflow_sees(
    workspace: dict[str, Path],
    cnn_harness: Callable[[list[str]], SimpleNamespace],
    device: str,
    gpus: list[str],
    message: str,
) -> None:
    state = cnn_harness(gpus)

    result = _invoke_cnn(workspace, device)

    assert result.exit_code != 0
    assert message in result.output
    assert len(state.built) == 1  # checked after loading, before any timing
    assert not workspace["results"].exists()


# ---------------------------------------------------------------- isolation --


def test_importing_the_benchmark_loads_no_model_dependencies() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, scripts.benchmark_latency; "
            "print(sorted(m for m in ('torch', 'transformers', 'tensorflow') if m in sys.modules))",
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    )
    assert result.stdout.strip() == "[]"


def test_help_lists_the_benchmark_options() -> None:
    result = CliRunner().invoke(benchmark_latency.main, ["--help"])
    assert result.exit_code == 0
    for option in (
        "--model",
        "--device",
        "--environment",
        "--num-images",
        "--batch-size",
        "--warmup-passes",
        "--measured-passes",
        "--cnn-weights",
    ):
        assert option in result.output
