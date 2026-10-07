"""Tests for the Phase 3 dashboard data export (TASK-017).

The exporter reads only committed results, so these tests use the real
``results/`` files: the committed JSON is regenerated and compared byte for
byte, and every failure case is a single edit to a copy of the real sources.
No model, network or TensorFlow is needed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import scripts.export_dashboard_data as export_dashboard_data
from click.testing import CliRunner, Result

from captioning.evaluation import (
    DashboardExportError,
    build_dashboard_data,
    render_dashboard_json,
)
from captioning.evaluation.latency import summarize

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"
DASHBOARD_JSON = REPO_ROOT / "frontend" / "src" / "generated" / "phase3-dashboard.json"
COMPARISON = RESULTS / "phase3-comparison" / "comparison.json"
SLICE_FINGERPRINT = "6b5628bfa410ed233ef9603beed3c05c9e63acebc8bfa31d7e63e634f9e25116"
MODELS = ["blip-base", "git-base-coco", "inceptionv3-transformer-stabilized", "vit-gpt2"]
BLIP_CPU = "phase3-latency-blip-base-greedy-cpu"  # sorts first: the baseline run
VIT_CUDA = "phase3-latency-vit-gpt2-greedy-cuda"
CNN = "inceptionv3-transformer-stabilized"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _committed() -> dict[str, Any]:
    return _load(DASHBOARD_JSON)


# ------------------------------------------------------------------ drift --


def test_committed_dashboard_data_matches_a_fresh_export() -> None:
    # read_text normalises line endings, so a CRLF checkout compares equal.
    committed = DASHBOARD_JSON.read_text(encoding="utf-8")
    fresh = render_dashboard_json(build_dashboard_data(RESULTS))
    assert committed == fresh, (
        f"{DASHBOARD_JSON.relative_to(REPO_ROOT).as_posix()} has drifted from results/; "
        "run python -m scripts.export_dashboard_data"
    )


# ------------------------------------------- committed file, checked raw --


def test_committed_data_lists_every_model_with_its_identity() -> None:
    data = _committed()
    assert data["schema_version"] == 1
    assert [m["model_id"] for m in data["models"]] == MODELS
    names = {m["model_id"]: m["display_name"] for m in data["models"]}
    assert names == {
        "blip-base": "BLIP-base",
        "git-base-coco": "GIT-base-coco",
        CNN: "CNN + Transformer (InceptionV3)",
        "vit-gpt2": "ViT-GPT2",
    }
    for model in data["models"]:
        latency = _load(RESULTS / model["latency"][0]["run_id"] / "latency.json")
        assert (model["hub_repo"], model["revision"]) == (latency["hub_repo"], latency["revision"])
        assert len(model["revision"]) == 40


def test_committed_data_states_the_slice_and_the_overlap_caveat() -> None:
    data, comparison = _committed(), _load(COMPARISON)
    assert data["overlap_caveat"] == comparison["caveat"]
    assert "not a held-out" in data["overlap_caveat"]
    s = data["slice"]
    assert (s["fingerprint_sha256"], s["images"], s["references"]) == (SLICE_FINGERPRINT, 500, 732)
    assert s["description"].startswith("500 COCO train2017 images")
    assert "first 32" in s["description"]


def test_committed_quality_values_are_the_comparison_values() -> None:
    data, comparison = _committed(), _load(COMPARISON)
    rows = {row["run_id"]: row for row in comparison["rows"]}
    exported = [q for m in data["models"] for q in m["quality"]]
    assert sorted(q["run_id"] for q in exported) == sorted(rows)
    for q in exported:
        row = rows[q["run_id"]]
        for key in ("kind", "revision", "decode_strategy", "decode_settings", "n_samples"):
            assert q[key] == row[key]
        assert q["metrics"] == row["metrics"]
        assert (RESULTS / q["run_id"]).is_dir()
    assert [m["key"] for m in data["quality"]["metrics"]] == comparison["metric_keys"]


def test_committed_latency_values_are_the_run_values() -> None:
    data = _committed()
    run_dirs = sorted(p.name for p in RESULTS.glob("phase3-latency-*"))
    exported = [entry for m in data["models"] for entry in m["latency"]]
    assert sorted(entry["run_id"] for entry in exported) == run_dirs
    assert len(run_dirs) == 8
    for model in data["models"]:
        assert [entry["device"] for entry in model["latency"]] == ["cpu", "cuda"]
        for entry in model["latency"]:
            run = _load(RESULTS / entry["run_id"] / "latency.json")
            assert run["model_id"] == model["model_id"]
            assert (entry["device"], entry["environment"]) == (run["device"], run["environment"])
            assert entry["load_seconds"] == run["load_seconds"]
            assert [b["batch_size"] for b in entry["batches"]] == [1, 8]
            for exported_batch, batch in zip(entry["batches"], run["batches"], strict=True):
                assert exported_batch["summary_seconds"] == batch["summary_seconds"]
                assert exported_batch["summary_seconds"] == summarize(batch["samples_seconds"])
            expected_mode = "sequential" if model["model_id"] == CNN else "batched"
            assert entry["batch_mode"] == expected_mode
        assert model["source_run_ids"] == [q["run_id"] for q in model["quality"]] + [
            entry["run_id"] for entry in model["latency"]
        ]


def test_committed_data_carries_the_latency_caveats() -> None:
    notes = " ".join(_committed()["latency"]["notes"])
    assert "not batched inference" in notes
    assert "not a controlled CPU-versus-GPU comparison" in notes
    assert "per image" in notes


# --------------------------------------------- failures on a copy of results --


@pytest.fixture
def results(tmp_path: Path) -> Path:
    """A copy of the exporter's inputs: the summary, its run ids, and the latency runs."""
    root = tmp_path / "results"
    shutil.copytree(RESULTS / "phase3-comparison", root / "phase3-comparison")
    for row in _load(COMPARISON)["rows"]:
        (root / row["run_id"]).mkdir()  # only the directory's existence is read
    for run_dir in RESULTS.glob("phase3-latency-*"):
        shutil.copytree(run_dir, root / run_dir.name)
    return root


def _edit(path: Path, change: Callable[[dict[str, Any]], None]) -> None:
    data = _load(path)
    change(data)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _edit_latency(root: Path, change: Callable[[dict[str, Any]], None], *names: str) -> None:
    for run_dir in [root / n for n in names] or sorted(root.glob("phase3-latency-*")):
        _edit(run_dir / "latency.json", change)


def test_a_copy_of_the_results_exports_the_committed_file(results: Path) -> None:
    text = render_dashboard_json(build_dashboard_data(results))
    assert text == DASHBOARD_JSON.read_text(encoding="utf-8")
    assert str(results.parent) not in text  # nothing machine-specific leaks in


def test_values_are_copied_verbatim(results: Path) -> None:
    def change(d: dict[str, Any]) -> None:
        d["rows"][0]["metrics"].update(bleu4=12.345678901234567, bleu3=7)

    _edit(results / "phase3-comparison" / "comparison.json", change)
    text = render_dashboard_json(build_dashboard_data(results))
    assert '"bleu4": 12.345678901234567' in text
    assert '"bleu3": 7,' in text  # an int stays an int


def _set(*keys: str, value: Any) -> Callable[[dict[str, Any]], None]:
    def change(d: dict[str, Any]) -> None:
        for key in keys[:-1]:
            d = d[key]
        d[keys[-1]] = value

    return change


def _first_batch(change: Callable[[dict[str, Any]], None]) -> Callable[[dict[str, Any]], None]:
    return lambda d: change(d["batches"][0])


def _rename(root: Path, old: str, new: str) -> None:
    (root / old).rename(root / new)


EDITS: dict[str, tuple[Callable[[Path], None], str]] = {
    "comparison missing": (
        lambda r: (r / "phase3-comparison" / "comparison.json").unlink(),
        "phase3-comparison: missing comparison.json",
    ),
    "comparison has an unknown field": (
        lambda r: _edit(r / "phase3-comparison" / "comparison.json", _set("extra", value=1)),
        "comparison.json is malformed",
    ),
    "a metric is stored as a string": (
        lambda r: _edit(
            r / "phase3-comparison" / "comparison.json",
            lambda d: d["rows"][0]["metrics"].update(bleu4="19.88"),
        ),
        "comparison.json is malformed",
    ),
    "comparison metric keys changed": (
        lambda r: _edit(
            r / "phase3-comparison" / "comparison.json", _set("metric_keys", value=["bleu4"])
        ),
        "metric_keys",
    ),
    "a comparison run directory is missing": (
        lambda r: (r / "stabilized-greedy").rmdir(),
        "source run stabilized-greedy has no directory",
    ),
    "no latency runs": (
        lambda r: [shutil.rmtree(p) for p in r.glob("phase3-latency-*")],
        "no latency runs found",
    ),
    "latency file is not JSON": (
        lambda r: (r / BLIP_CPU / "latency.json").write_text("{", encoding="utf-8"),
        f"{BLIP_CPU}: latency.json is malformed",
    ),
    "directory doesn't name the run": (
        lambda r: _rename(r, BLIP_CPU, "phase3-latency-blip-base-greedy-tpu"),
        f"latency.json describes {BLIP_CPU}",
    ),
    "unknown captioner": (
        lambda r: _edit_latency(r, _set("captioner", value="OtherCaptioner"), BLIP_CPU),
        "unknown captioner 'OtherCaptioner'",
    ),
    "settings differ between runs": (
        lambda r: _edit_latency(r, _set("settings", "warmup_passes", value=2), VIT_CUDA),
        f"{VIT_CUDA}: settings doesn't match {BLIP_CPU}",
    ),
    "inputs differ between runs": (
        lambda r: _edit_latency(r, lambda d: d["inputs"]["images"].reverse(), VIT_CUDA),
        f"{VIT_CUDA}: inputs doesn't match {BLIP_CPU}",
    ),
    "latency slice differs from the quality slice": (
        lambda r: _edit_latency(r, _set("inputs", "slice", "fingerprint_sha256", value="0" * 64)),
        "slice fingerprint",
    ),
    "a sample is missing": (
        lambda r: _edit_latency(r, _first_batch(lambda b: b["samples_seconds"].pop()), BLIP_CPU),
        "has 159 samples, expected 160",
    ),
    "summary doesn't match the samples": (
        lambda r: _edit_latency(
            r, _first_batch(_set("summary_seconds", "median", value=0.5)), BLIP_CPU
        ),
        "summary_seconds doesn't match its samples_seconds",
    ),
    "runs disagree on the revision": (
        lambda r: _edit_latency(r, _set("revision", value="f" * 40), BLIP_CPU),
        "blip-base: runs disagree on revision",
    ),
    "no run records the revision": (
        lambda r: _edit_latency(
            r,
            _set("revision", value=None),
            f"phase3-latency-{CNN}-greedy-cpu",
            f"phase3-latency-{CNN}-greedy-cuda",
        ),
        f"{CNN}: no run records its revision",
    ),
    "a model has no display name": (
        lambda r: _edit(
            r / "phase3-comparison" / "comparison.json",
            lambda d: d["rows"][0].update(model_id="new-model"),
        ),
        "no display name for model(s) ['new-model']",
    ),
}


@pytest.mark.parametrize("case", list(EDITS))
def test_inconsistent_results_are_refused(results: Path, case: str) -> None:
    edit, message = EDITS[case]
    edit(results)
    with pytest.raises(DashboardExportError) as excinfo:
        build_dashboard_data(results)
    assert message in str(excinfo.value)


# --------------------------------------------------------------------- CLI --


def _invoke(*args: str) -> Result:
    return CliRunner().invoke(export_dashboard_data.main, list(args))


def test_check_passes_on_the_committed_file() -> None:
    result = _invoke("--results-root", str(RESULTS), "--output", str(DASHBOARD_JSON), "--check")
    assert result.exit_code == 0, result.output
    assert "is up to date (4 models, 13 runs)" in result.output


def test_export_writes_lf_utf8_that_then_checks_clean(results: Path, tmp_path: Path) -> None:
    output = tmp_path / "data" / "dashboard.json"
    result = _invoke("--results-root", str(results), "--output", str(output))
    assert result.exit_code == 0, result.output
    raw = output.read_bytes()
    assert b"\r" not in raw
    assert raw.endswith(b"}\n")
    assert "§".encode() in raw  # not escaped
    assert raw.decode("utf-8") == DASHBOARD_JSON.read_text(encoding="utf-8")
    check = _invoke("--results-root", str(results), "--output", str(output), "--check")
    assert check.exit_code == 0, check.output


@pytest.mark.parametrize("stale", ["edited", "missing"])
def test_check_fails_on_a_stale_or_missing_file(results: Path, tmp_path: Path, stale: str) -> None:
    output = tmp_path / "dashboard.json"
    if stale == "edited":
        text = DASHBOARD_JSON.read_text(encoding="utf-8")
        output.write_text(text.replace("56.60798221374851", "56.6"), encoding="utf-8")
    before = output.read_bytes() if output.exists() else None
    result = _invoke("--results-root", str(results), "--output", str(output), "--check")
    assert result.exit_code == 1
    assert "is out of date" in result.output
    assert (output.read_bytes() if output.exists() else None) == before


def test_an_export_error_writes_nothing(results: Path, tmp_path: Path) -> None:
    _edit_latency(results, _set("revision", value="f" * 40), BLIP_CPU)
    output = tmp_path / "dashboard.json"
    result = _invoke("--results-root", str(results), "--output", str(output))
    assert result.exit_code == 1
    assert "runs disagree on revision" in result.output
    assert not output.exists()


def test_help_lists_the_export_options() -> None:
    result = _invoke("--help")
    assert result.exit_code == 0
    for option in ("--results-root", "--comparison-id", "--latency-prefix", "--output", "--check"):
        assert option in result.output


def test_importing_the_exporter_loads_no_model_dependencies() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, scripts.export_dashboard_data; "
            "print(sorted(m for m in ('torch', 'transformers', 'tensorflow') if m in sys.modules))",
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    )
    assert result.stdout.strip() == "[]"
