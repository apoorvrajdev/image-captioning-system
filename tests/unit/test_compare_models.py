"""Tests for the Phase 3 comparison runner (TASK-012).

Offline: captioners are fakes built on the real :class:`HFCaptioner` identity,
so no model is downloaded or run and ``torch`` / ``transformers`` are never
imported. The fixture slice is three rows written to ``tmp_path``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import scripts.compare_models as compare_models
import yaml
from click.testing import CliRunner, Result

from captioning.baselines import Captioner, CNNCaptioner, HFCaptioner
from captioning.config import AppConfig, BaselineDecodeConfig, ComparedModelConfig
from captioning.evaluation import load_eval_slice
from captioning.evaluation.tokenization import strip_sentinels
from captioning.preprocessing.caption import preprocess_caption

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMITTED_SLICES = [
    REPO_ROOT / "results" / "stabilized-greedy" / "predictions.jsonl",
    REPO_ROOT / "results" / "stabilized-beam-w4-lp07-rp12" / "predictions.jsonl",
]
# Content fingerprint of the frozen slice (EVAL_METHODOLOGY.md § 8.2).
PROTOCOL_SLICE_FINGERPRINT = "6b5628bfa410ed233ef9603beed3c05c9e63acebc8bfa31d7e63e634f9e25116"
STANDARD_FILES = {
    "metrics.json",
    "predictions.jsonl",
    "diagnostics.jsonl",
    "run_meta.json",
    "report.md",
}
FIXTURE_ROWS = [
    {
        "image": "/kaggle/coco2017/train2017/img0.jpg",
        "prediction": "x",
        "references": ["[start] a dog on the grass [end]", "[start] a brown dog [end]"],
    },
    {
        "image": "/kaggle/coco2017/train2017/img1.jpg",
        "prediction": "x",
        "references": ["[start] a red car parked [end]"],
    },
    {
        "image": "/kaggle/coco2017/train2017/img2.jpg",
        "prediction": "x",
        "references": [
            "[start] two cats sleeping [end]",
            "[start] cats on a bed [end]",
            "[start] a pair of cats [end]",
        ],
    },
]


class FakeHFCaptioner(HFCaptioner):
    """Real HF identity and decode settings; scripted captions, no model."""

    def __init__(
        self, model: ComparedModelConfig, decode: BaselineDecodeConfig, *, device: str = "cpu"
    ) -> None:
        super().__init__(model, decode, device=device)
        self.loads = 0
        self.batches: list[list[str]] = []

    def load(self) -> None:
        self.loads += 1

    def _raw_captions(self, image_paths: list[Path]) -> list[str]:
        self.batches.append([p.name for p in image_paths])
        return [f"A {self.identity.model_id} Caption, for {p.stem}." for p in image_paths]


@pytest.fixture
def workspace(tmp_path: Path) -> dict[str, Path]:
    slice_path = tmp_path / "slice" / "predictions.jsonl"
    slice_path.parent.mkdir()
    slice_path.write_text("".join(json.dumps(r) + "\n" for r in FIXTURE_ROWS), encoding="utf-8")
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    for row in FIXTURE_ROWS:
        (images_dir / Path(row["image"]).name).write_bytes(b"")  # existence only; fakes don't read
    return {"slice": slice_path, "images": images_dir, "results": tmp_path / "results"}


@pytest.fixture
def fake_captioners(monkeypatch: pytest.MonkeyPatch) -> list[FakeHFCaptioner]:
    built: list[FakeHFCaptioner] = []

    def build(spec: compare_models.ModelSpec, config: AppConfig, **_: Any) -> Captioner:
        captioner = FakeHFCaptioner(spec.model, config.compare.baseline_decode)
        built.append(captioner)
        return captioner

    monkeypatch.setattr(compare_models, "build_captioner", build)
    return built


def _invoke(workspace: dict[str, Path], *args: str, config: Path | None = None) -> Result:
    base = [
        "--config",
        str(config or REPO_ROOT / "configs" / "base.yaml"),
        "--slice",
        str(workspace["slice"]),
        "--images-dir",
        str(workspace["images"]),
        "--results-root",
        str(workspace["results"]),
        "--expected-images",
        "3",
        "--expected-references",
        "6",
        "--skip-meteor",
    ]
    return CliRunner().invoke(compare_models.main, [*base, *args])


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_writes_one_contract_directory_per_model(
    workspace: dict[str, Path], fake_captioners: list[FakeHFCaptioner]
) -> None:
    result = _invoke(
        workspace, "--model", "blip-base", "--model", "git-base-coco", "--batch-size", "2"
    )
    assert result.exit_code == 0, result.output

    compare = AppConfig().compare
    eval_slice = load_eval_slice(workspace["slice"], workspace["images"])
    images = [str(p) for p in eval_slice.image_paths]
    run_dirs = sorted(p.name for p in workspace["results"].iterdir())
    assert run_dirs == ["phase3-blip-base-greedy", "phase3-git-base-coco-greedy"]

    for captioner, model in zip(
        fake_captioners, [compare.baselines[0], compare.baselines[2]], strict=True
    ):
        run_dir = workspace["results"] / f"phase3-{model.model_id}-greedy"
        assert {p.name for p in run_dir.iterdir()} == STANDARD_FILES | {"comparison_meta.json"}
        assert captioner.loads == 1
        assert captioner.batches == [["img0.jpg", "img1.jpg"], ["img2.jpg"]]

        rows = _read_jsonl(run_dir / "predictions.jsonl")
        assert [r["image"] for r in rows] == images
        assert [r["references"] for r in rows] == [row["references"] for row in FIXTURE_ROWS]
        assert [r["prediction"] for r in rows] == [
            strip_sentinels(preprocess_caption(f"A {model.model_id} Caption, for img{i}."))
            for i in range(3)
        ]

        run_meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
        pinned = f"{model.hub_repo}@{model.revision}"
        assert {k: v for k, v in run_meta.items() if k != "timestamp_utc"} == {
            "model_id": model.model_id,
            "decode_strategy": "greedy",
            "weights_path": pinned,
            "tokenizer_dir": pinned,
            "n_samples": 3,
            "max_length": 40,
            "beam_width": None,
            "length_penalty": None,
            "repetition_penalty": 1.0,
        }
        assert json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))["n_examples"] == 3

        meta = json.loads((run_dir / "comparison_meta.json").read_text(encoding="utf-8"))
        assert meta == {
            "protocol": "docs/EVAL_METHODOLOGY.md § 8",
            "model_id": model.model_id,
            "backend": "huggingface",
            "captioner": "FakeHFCaptioner",
            "hub_repo": model.hub_repo,
            "revision": model.revision,
            "decode_settings": compare.baseline_decode.model_dump(),
            "normalisation": "preprocess_caption -> strip_sentinels",
            "slice": {
                "source": workspace["slice"].as_posix(),
                "fingerprint_sha256": compare_models.slice_fingerprint(eval_slice),
                "images": 3,
                "references": 6,
            },
            "batch_size": 2,
            "device": "cpu",
            "seed": 42,
            "metrics": {"meteor": False, "cider": True},
        }


def test_committed_slice_is_the_protocol_default() -> None:
    defaults = {p.name: p.default for p in compare_models.main.params}
    assert Path(str(defaults["slice_path"])) == Path("results/stabilized-greedy/predictions.jsonl")
    assert (defaults["expected_images"], defaults["expected_references"]) == (500, 732)

    for path in COMMITTED_SLICES:
        eval_slice = load_eval_slice(path, "images")
        assert len(eval_slice) == 500
        assert sum(len(refs) for refs in eval_slice.references) == 732
        assert compare_models.slice_fingerprint(eval_slice) == PROTOCOL_SLICE_FINGERPRINT


def _assert_nothing_ran(
    result: Result, workspace: dict[str, Path], built: list[FakeHFCaptioner], message: str
) -> None:
    assert result.exit_code != 0
    assert message in result.output
    assert built == []
    assert not any(workspace["results"].glob("phase3-*"))


def test_slice_counts_must_match_the_expected_protocol(
    workspace: dict[str, Path], fake_captioners: list[FakeHFCaptioner]
) -> None:
    result = _invoke(workspace, "--model", "blip-base", "--expected-images", "500")
    _assert_nothing_ran(result, workspace, fake_captioners, "expected 500 images and 6 references")


def test_unknown_and_duplicate_models_are_rejected(
    workspace: dict[str, Path], fake_captioners: list[FakeHFCaptioner]
) -> None:
    unknown = _invoke(workspace, "--model", "blip-large")
    _assert_nothing_ran(unknown, workspace, fake_captioners, "unknown model id 'blip-large'")
    assert "vit-gpt2" in unknown.output

    duplicate = _invoke(workspace, "--model", "vit-gpt2", "--model", "vit-gpt2")
    _assert_nothing_ran(duplicate, workspace, fake_captioners, "selected more than once")


def test_cnn_needs_its_checkpoint(
    workspace: dict[str, Path], fake_captioners: list[FakeHFCaptioner]
) -> None:
    result = _invoke(workspace, "--model", "inceptionv3-transformer-stabilized")
    _assert_nothing_ran(result, workspace, fake_captioners, "--cnn-weights and --cnn-tokenizer-dir")


def test_sampling_decode_settings_are_rejected(
    workspace: dict[str, Path], fake_captioners: list[FakeHFCaptioner], tmp_path: Path
) -> None:
    config = tmp_path / "sampling.yaml"
    config.write_text(yaml.safe_dump({"compare": {"baseline_decode": {"do_sample": True}}}))

    result = _invoke(workspace, "--model", "blip-base", config=config)
    _assert_nothing_ran(result, workspace, fake_captioners, "sampling")


def test_missing_images_fail_before_any_model_loads(
    workspace: dict[str, Path], fake_captioners: list[FakeHFCaptioner]
) -> None:
    (workspace["images"] / "img1.jpg").unlink()

    result = _invoke(workspace, "--model", "blip-base")
    _assert_nothing_ran(result, workspace, fake_captioners, "1 of 3 slice images are missing")
    assert "img1.jpg" in result.output


def test_existing_run_directory_is_never_overwritten(
    workspace: dict[str, Path], fake_captioners: list[FakeHFCaptioner]
) -> None:
    existing = workspace["results"] / "phase3-vit-gpt2-greedy"
    existing.mkdir(parents=True)
    (existing / "metrics.json").write_text("keep me", encoding="utf-8")

    result = _invoke(workspace, "--model", "blip-base", "--model", "vit-gpt2")

    assert result.exit_code != 0
    assert "already exists" in result.output
    assert fake_captioners == []
    assert not (workspace["results"] / "phase3-blip-base-greedy").exists()
    assert (existing / "metrics.json").read_text(encoding="utf-8") == "keep me"


def test_build_captioner_propagates_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = AppConfig()
    [hf_spec, cnn_spec] = compare_models.resolve_models(
        config, ["vit-gpt2", "inceptionv3-transformer-stabilized"], tmp_path, "p-"
    )
    assert (hf_spec.backend, hf_spec.run_dir) == ("huggingface", tmp_path / "p-vit-gpt2-greedy")
    assert (cnn_spec.backend, cnn_spec.run_dir.name) == (
        "cnn",
        "p-inceptionv3-transformer-stabilized-greedy",
    )

    hf = compare_models.build_captioner(
        hf_spec, config, device="cuda", cnn_weights=None, cnn_tokenizer_dir=None
    )
    assert isinstance(hf, HFCaptioner)
    assert (hf.identity.model_id, hf.identity.revision) == (
        "vit-gpt2",
        config.compare.baselines[1].revision,
    )
    assert hf.generate_kwargs["max_new_tokens"] == 40
    assert hf._device == "cuda"

    calls: list[tuple[Any, ...]] = []

    def fake_from_artifacts(weights: Path, tokenizer_dir: Path, cfg: AppConfig) -> str:
        calls.append((weights, tokenizer_dir, cfg))
        return "cnn-captioner"

    monkeypatch.setattr(CNNCaptioner, "from_artifacts", fake_from_artifacts)
    built = compare_models.build_captioner(
        cnn_spec, config, device="cpu", cnn_weights=Path("w.h5"), cnn_tokenizer_dir=Path("tok")
    )
    assert built == "cnn-captioner"
    assert calls == [(Path("w.h5"), Path("tok"), config)]


def test_importing_the_runner_loads_no_model_dependencies() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, scripts.compare_models; "
            "print(sorted(m for m in ('torch', 'transformers', 'tensorflow') if m in sys.modules))",
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    )
    assert result.stdout.strip() == "[]"


def test_help_lists_the_runner_options() -> None:
    result = CliRunner().invoke(compare_models.main, ["--help"])
    assert result.exit_code == 0
    for option in ("--model", "--images-dir", "--slice", "--cnn-weights", "--batch-size"):
        assert option in result.output
