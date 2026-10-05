"""Tests for the Phase 3 evaluation-slice loader (TASK-010).

The slice is read from committed ``results/<run_id>/predictions.jsonl`` files
(``docs/EVAL_METHODOLOGY.md`` § 8.2). Offline: no COCO images are needed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from captioning.evaluation import EvalSlice, load_eval_slice

REPO_ROOT = Path(__file__).resolve().parents[2]
GREEDY = REPO_ROOT / "results" / "stabilized-greedy" / "predictions.jsonl"
BEAM = REPO_ROOT / "results" / "stabilized-beam-w4-lp07-rp12" / "predictions.jsonl"
IMAGES_DIR = Path("coco2017") / "train2017"


def _raw_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_greedy_slice_matches_the_committed_file() -> None:
    eval_slice = load_eval_slice(GREEDY, IMAGES_DIR)
    rows = _raw_rows(GREEDY)

    assert len(eval_slice) == len(rows) == 500
    assert eval_slice.image_paths == tuple(
        IMAGES_DIR / PurePosixPath(str(row["image"])).name for row in rows
    )
    assert eval_slice.references == tuple(tuple(row["references"]) for row in rows)
    assert sum(len(refs) for refs in eval_slice.references) == 732
    assert eval_slice.source == GREEDY


def test_committed_greedy_and_beam_runs_share_the_same_slice() -> None:
    greedy = load_eval_slice(GREEDY, IMAGES_DIR)
    beam = load_eval_slice(BEAM, IMAGES_DIR)

    assert greedy == beam
    assert greedy.source != beam.source


def test_loader_does_not_require_image_files(tmp_path: Path) -> None:
    missing_dir = tmp_path / "not-created"
    eval_slice = load_eval_slice(GREEDY, missing_dir)

    assert len(eval_slice) == 500
    assert not missing_dir.exists()


def test_windows_style_stored_paths_are_remapped_by_file_name(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    rows = [
        {"image": "C:\\data\\train2017\\000000000001.jpg", "prediction": "a", "references": ["x"]},
        {
            "image": "/kaggle/train2017/000000000002.jpg",
            "prediction": "b",
            "references": ["y", "z"],
        },
    ]
    predictions.write_text("".join(json.dumps(r) + "\n" for r in rows) + "\n", encoding="utf-8")

    eval_slice = load_eval_slice(predictions, tmp_path / "images")

    assert eval_slice == EvalSlice(
        source=predictions,
        image_paths=(
            tmp_path / "images" / "000000000001.jpg",
            tmp_path / "images" / "000000000002.jpg",
        ),
        references=(("x",), ("y", "z")),
    )


@pytest.mark.parametrize(
    "row",
    [
        {"prediction": "a", "references": ["x"]},
        {"image": "a.jpg", "prediction": "a"},
        {"image": "a.jpg", "prediction": "a", "references": "x"},
        {"image": "a.jpg", "prediction": "a", "references": []},
        {"image": "", "prediction": "a", "references": ["x"]},
    ],
)
def test_malformed_rows_raise_with_the_line_number(tmp_path: Path, row: dict[str, object]) -> None:
    predictions = tmp_path / "predictions.jsonl"
    good = {"image": "ok.jpg", "prediction": "b", "references": ["y"]}
    predictions.write_text(json.dumps(good) + "\n" + json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r":2:"):
        load_eval_slice(predictions, tmp_path)


def test_loader_imports_no_model_dependencies() -> None:
    code = (
        "import sys, captioning.evaluation.slice; "
        "print(sorted(m for m in ('tensorflow', 'transformers', 'torch') if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"
