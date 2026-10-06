"""Tests for the Phase 3 cross-run comparison summary (TASK-013).

Run directories are produced by the real TASK-012 runner with the fake
captioner from ``test_compare_models``, so the summary is tested against the
exact artefacts the runner writes. No model is loaded or downloaded.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import scripts.compare_models as compare_models
import scripts.compare_runs as compare_runs
from click.testing import CliRunner, Result

from captioning.config import AppConfig
from tests.unit.test_compare_models import (
    FIXTURE_ROWS,
    PROTOCOL_SLICE_FINGERPRINT,
    FakeHFCaptioner,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMITTED = [
    REPO_ROOT / "results" / name for name in ("stabilized-greedy", "stabilized-beam-w4-lp07-rp12")
]
METRIC_KEYS = ["bleu1", "bleu2", "bleu3", "bleu4", "rouge_l", "meteor", "cider"]
BLIP, GIT, VIT = "phase3-blip-base-greedy", "phase3-git-base-coco-greedy", "phase3-vit-gpt2-greedy"


def _make_runs(base: Path, rows: list[dict[str, Any]], models: list[str]) -> None:
    """Write a slice, its images, and one runner output directory per model."""
    base.mkdir(parents=True)
    slice_path = base / "predictions.jsonl"
    slice_path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    images = base / "images"
    images.mkdir()
    for row in rows:
        (images / Path(row["image"]).name).write_bytes(b"")
    n_refs = sum(len(r["references"]) for r in rows)
    args = [
        "--config", str(REPO_ROOT / "configs" / "base.yaml"),
        "--slice", str(slice_path),
        "--images-dir", str(images),
        "--results-root", str(base / "results"),
        "--expected-images", str(len(rows)),
        "--expected-references", str(n_refs),
        "--skip-meteor",
    ]  # fmt: skip
    for model in models:
        args += ["--model", model]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            compare_models,
            "build_captioner",
            lambda spec, config, **_: FakeHFCaptioner(spec.model, config.compare.baseline_decode),
        )
        result = CliRunner().invoke(compare_models.main, args)
    assert result.exit_code == 0, result.output


@pytest.fixture(scope="module")
def template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("runs")
    _make_runs(root / "same", FIXTURE_ROWS, ["blip-base", "git-base-coco"])
    _make_runs(root / "other", FIXTURE_ROWS[:2], ["vit-gpt2"])  # a different, smaller slice
    return root


@pytest.fixture
def runs(template: Path, tmp_path: Path) -> Path:
    shutil.copytree(template, tmp_path / "runs")
    return tmp_path / "runs" / "same" / "results"


def _summarise(*args: str | Path, output: Path) -> Result:
    return CliRunner().invoke(
        compare_runs.main, [*(str(a) for a in args), "--output-dir", str(output)]
    )


def _edit_json(path: Path, edit: Callable[[dict[str, Any]], None]) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    edit(data)
    path.write_text(json.dumps(data), encoding="utf-8")


def _rejected(result: Result, output: Path, *fragments: str) -> None:
    assert result.exit_code != 0
    for fragment in fragments:
        assert fragment in result.output
    assert not output.exists()  # nothing is written on a mismatch


def test_compatible_runs_produce_a_summary(runs: Path, tmp_path: Path) -> None:
    output = tmp_path / "summary"
    result = _summarise(runs / GIT, runs / BLIP, output=output)
    assert result.exit_code == 0, result.output

    summary = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    meta = json.loads((runs / BLIP / "comparison_meta.json").read_text(encoding="utf-8"))
    assert summary["protocol"] == "docs/EVAL_METHODOLOGY.md § 8"
    assert summary["normalisation"] == "preprocess_caption -> strip_sentinels"
    assert summary["slice"] == {
        "fingerprint_sha256": meta["slice"]["fingerprint_sha256"],
        "images": 3,
        "references": 6,
        "references_per_image": 2.0,
    }
    assert "§ 8.5" in summary["caveat"]
    assert summary["metric_keys"] == METRIC_KEYS

    compare = AppConfig().compare
    assert [row["run_id"] for row in summary["rows"]] == [BLIP, GIT]  # sorted by model id
    for row, model in zip(
        summary["rows"], [compare.baselines[0], compare.baselines[2]], strict=True
    ):
        metrics = json.loads((runs / row["run_id"] / "metrics.json").read_text(encoding="utf-8"))
        assert row == {
            "run_id": f"phase3-{model.model_id}-greedy",
            "kind": "phase3",
            "model_id": model.model_id,
            "backend": "huggingface",
            "captioner": "FakeHFCaptioner",
            "hub_repo": model.hub_repo,
            "revision": model.revision,
            "weights_path": f"{model.hub_repo}@{model.revision}",
            "decode_strategy": "greedy",
            "decode_settings": compare.baseline_decode.model_dump(),
            "n_samples": 3,
            "metrics": {key: metrics[key] for key in METRIC_KEYS},  # verbatim
            "metric_errors": metrics["errors"],
        }

    markdown = (output / "comparison.md").read_text(encoding="utf-8")
    assert "3 images, 6 references" in markdown
    assert BLIP in markdown and GIT in markdown and "§ 8.5" in markdown


def test_summary_does_not_depend_on_argument_order(runs: Path, tmp_path: Path) -> None:
    assert _summarise(runs / BLIP, runs / GIT, output=tmp_path / "a").exit_code == 0
    assert _summarise(runs / GIT, runs / BLIP, output=tmp_path / "b").exit_code == 0
    for name in ("comparison.json", "comparison.md"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_committed_runs_are_accepted_as_reference_rows(tmp_path: Path) -> None:
    output = tmp_path / "summary"
    result = _summarise(
        "--reference-run", COMMITTED[1], "--reference-run", COMMITTED[0], output=output
    )
    assert result.exit_code == 0, result.output

    summary = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    assert summary["protocol"] is None  # pre-harness runs record no protocol
    assert summary["slice"]["fingerprint_sha256"] == PROTOCOL_SLICE_FINGERPRINT
    assert (summary["slice"]["images"], summary["slice"]["references"]) == (500, 732)
    assert [(r["run_id"], r["kind"], r["decode_strategy"]) for r in summary["rows"]] == [
        ("stabilized-beam-w4-lp07-rp12", "reference", "beam"),
        ("stabilized-greedy", "reference", "greedy"),
    ]
    for row in summary["rows"]:
        assert (row["backend"], row["hub_repo"], row["revision"]) == (None, None, None)
        metrics = json.loads((REPO_ROOT / "results" / row["run_id"] / "metrics.json").read_text())
        assert row["metrics"] == {key: metrics[key] for key in METRIC_KEYS}


@pytest.mark.parametrize(
    ("edit", "fragments"),
    [
        (lambda m: m["slice"].update(fingerprint_sha256="0" * 64), ["slice fingerprint"]),
        (lambda m: m["slice"].update(images=4), ["slice.images is 4"]),
        (lambda m: m["slice"].update(references=7), ["slice.references is 7"]),
        (lambda m: m.update(protocol="docs/OTHER.md § 1"), ["protocol"]),
        (lambda m: m.update(normalisation="lowercase"), ["normalisation"]),
        (lambda m: m.pop("revision"), ["comparison_meta.json is malformed"]),
    ],
)
def test_inconsistent_comparison_meta_is_rejected(
    runs: Path, tmp_path: Path, edit: Callable[[dict[str, Any]], None], fragments: list[str]
) -> None:
    _edit_json(runs / GIT / "comparison_meta.json", edit)
    output = tmp_path / "summary"
    _rejected(_summarise(runs / BLIP, runs / GIT, output=output), output, GIT, *fragments)


def test_altered_predictions_change_the_fingerprint_and_are_rejected(
    runs: Path, tmp_path: Path
) -> None:
    path = runs / GIT / "predictions.jsonl"
    path.write_text(
        path.read_text(encoding="utf-8").replace("a red car", "a blue car"), encoding="utf-8"
    )
    output = tmp_path / "summary"
    _rejected(_summarise(runs / BLIP, runs / GIT, output=output), output, GIT, "slice fingerprint")


def test_runs_on_a_different_slice_are_rejected(runs: Path, tmp_path: Path) -> None:
    other = runs.parents[1] / "other" / "results" / VIT
    output = tmp_path / "summary"
    _rejected(_summarise(runs / BLIP, other, output=output), output, VIT, "image count 2", BLIP)


@pytest.mark.parametrize(
    ("damage", "fragments"),
    [
        (lambda d: (d / "comparison_meta.json").unlink(), ["missing comparison_meta.json"]),
        (lambda d: (d / "metrics.json").unlink(), ["missing metrics.json"]),
        (
            lambda d: (d / "comparison_meta.json").write_text("{not json"),
            ["comparison_meta.json is malformed"],
        ),
        (
            lambda d: _edit_json(d / "metrics.json", lambda m: m.pop("bleu4")),
            ["metrics.json is malformed"],
        ),
        (
            lambda d: _edit_json(d / "run_meta.json", lambda m: m.update(n_samples=2)),
            ["n_samples is 2"],
        ),
    ],
)
def test_incomplete_or_malformed_run_directories_are_rejected(
    runs: Path, tmp_path: Path, damage: Callable[[Path], object], fragments: list[str]
) -> None:
    damage(runs / GIT)
    output = tmp_path / "summary"
    _rejected(_summarise(runs / BLIP, runs / GIT, output=output), output, GIT, *fragments)


def test_duplicate_model_identities_are_rejected(runs: Path, tmp_path: Path) -> None:
    shutil.copytree(runs / BLIP, runs / "blip-again")
    output = tmp_path / "summary"
    _rejected(
        _summarise(runs / BLIP, runs / "blip-again", output=output),
        output,
        "blip-again",
        "duplicates",
    )


def test_existing_summary_is_never_overwritten(runs: Path, tmp_path: Path) -> None:
    output = tmp_path / "summary"
    output.mkdir()
    (output / "comparison.json").write_text("keep me", encoding="utf-8")

    result = _summarise(runs / BLIP, runs / GIT, output=output)

    assert result.exit_code != 0
    assert "already exists" in result.output
    assert (output / "comparison.json").read_text(encoding="utf-8") == "keep me"
