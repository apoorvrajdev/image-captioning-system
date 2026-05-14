"""Tests for ROUGE-L, CIDEr, METEOR adapters and the unified runner.

We don't validate the upstream implementations — they have their own test
suites. We *do* validate our adapters: sentinel stripping, ragged references,
the perfect-prediction bound, and that the unified ``compute_all_metrics``
correctly records partial failures in ``errors`` rather than crashing the
whole pass.
"""

from __future__ import annotations

import json

import pytest

from captioning.evaluation import (
    MIN_SAMPLES_FOR_CIDER,
    BleuBreakdown,
    RunMeta,
    compute_all_metrics,
    corpus_bleu_breakdown,
    corpus_bleu_score,
    corpus_cider_score,
    corpus_rouge_l_score,
    diagnose_many,
    diagnose_sample,
    write_diagnostics_jsonl,
    write_run_artifacts,
)

# ---- BLEU ------------------------------------------------------------------


def test_bleu_breakdown_returns_all_four_orders() -> None:
    refs = [["a man riding a bike"], ["a dog in the park"]]
    preds = ["a man riding a bike", "a dog in the park"]
    result = corpus_bleu_breakdown(preds, refs)
    assert isinstance(result, BleuBreakdown)
    assert result.bleu1 == pytest.approx(100.0)
    assert result.bleu2 == pytest.approx(100.0)
    assert result.bleu4 == pytest.approx(100.0)
    assert corpus_bleu_score(preds, refs) == pytest.approx(result.bleu4)


def test_bleu_strips_sentinels_before_scoring() -> None:
    refs = [["[start] a man riding a bike [end]"]]
    preds = ["[start] a man riding a bike [end]"]
    assert corpus_bleu_score(preds, refs) == pytest.approx(100.0)


# ---- ROUGE-L ---------------------------------------------------------------


def test_rouge_l_perfect_matches_score_100() -> None:
    refs = [["a man riding a bike"], ["a dog in the park"]]
    preds = ["a man riding a bike", "a dog in the park"]
    score = corpus_rouge_l_score(preds, refs)
    assert score == pytest.approx(100.0)


def test_rouge_l_partial_overlap_scores_in_range() -> None:
    refs = [["a man riding a bike on a road"]]
    preds = ["a man on a road"]
    score = corpus_rouge_l_score(preds, refs)
    # Reference has 7 tokens, prediction has 5 tokens, LCS=5
    # P = 5/5 = 1.0, R = 5/7 ≈ 0.71, F ≈ 0.83
    assert 70.0 < score < 90.0


def test_rouge_l_picks_best_reference() -> None:
    refs = [["xyz qrs nothing matches", "a man riding a bike"]]
    preds = ["a man riding a bike"]
    score = corpus_rouge_l_score(preds, refs)
    assert score == pytest.approx(100.0)


def test_rouge_l_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        corpus_rouge_l_score(["a"], [["a"], ["b"]])


# ---- CIDEr -----------------------------------------------------------------


def test_cider_requires_minimum_samples() -> None:
    with pytest.raises(ValueError, match="degenerate"):
        corpus_cider_score(["a man"], [["a man"]])


def test_cider_returns_positive_for_good_predictions() -> None:
    refs = [
        ["a man riding a bike"],
        ["a dog in the park"],
        ["two children playing"],
    ]
    preds = ["a man riding a bike", "a dog in the park", "two children playing"]
    score = corpus_cider_score(preds, refs)
    assert score > 0.0


# ---- Runner ----------------------------------------------------------------


def test_compute_all_metrics_returns_every_field() -> None:
    refs = [["a man riding a bike"], ["a dog in the park"]]
    preds = ["a man riding a bike", "a dog in the park"]
    report = compute_all_metrics(preds, refs, include_meteor=False, include_cider=False)
    assert report.n_examples == 2
    assert report.bleu1 is not None
    assert report.bleu4 is not None
    assert report.rouge_l is not None
    assert report.meteor is None  # explicitly skipped
    assert report.cider is None  # explicitly skipped


def test_compute_all_metrics_skips_cider_on_tiny_corpus() -> None:
    refs = [["a man riding a bike"]]
    preds = ["a man riding a bike"]
    report = compute_all_metrics(preds, refs, include_meteor=False)
    assert report.cider is None
    assert "cider" in report.errors


def test_compute_all_metrics_serialises_to_dict() -> None:
    refs = [["a man riding a bike"], ["a dog in the park"]]
    preds = ["a man riding a bike", "a dog in the park"]
    report = compute_all_metrics(preds, refs, include_meteor=False, include_cider=False)
    payload = report.to_dict()
    # JSON-roundtrip must not lose information.
    assert json.loads(json.dumps(payload)) == payload


# ---- Inspection -----------------------------------------------------------


def test_diagnose_sample_flags_empty_prediction() -> None:
    d = diagnose_sample("img.jpg", "", ["a man riding a bike"])
    assert "empty" in d.flags
    assert d.length_tokens == 0


def test_diagnose_sample_flags_repetitive_prediction() -> None:
    d = diagnose_sample("img.jpg", "a a a a man", ["a man riding a bike"])
    assert "repetitive" in d.flags
    assert d.longest_repeat_run == 4


def test_diagnose_sample_flags_very_short_prediction() -> None:
    d = diagnose_sample("img.jpg", "a man", ["a man riding a bike"])
    assert "very_short" in d.flags


def test_diagnose_many_writes_jsonl(tmp_path) -> None:
    images = ["a.jpg", "b.jpg"]
    preds = ["a man riding a bike", ""]
    refs = [["a man on a bicycle"], ["a dog in the park"]]
    diags = diagnose_many(images, preds, refs)
    out = tmp_path / "diag.jsonl"
    write_diagnostics_jsonl(diags, out)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["image"] == "a.jpg"
    # Empty prediction also flags as ``very_short`` because it has 0 tokens.
    assert "empty" in parsed[1]["flags"]


# ---- Benchmark scaffolding ------------------------------------------------


def test_write_run_artifacts_emits_expected_files(tmp_path) -> None:
    images = ["a.jpg", "b.jpg"]
    preds = ["a man riding a bike", "a dog in the park"]
    refs = [["a man on a bicycle"], ["a dog in the park"]]
    diags = diagnose_many(images, preds, refs)
    report = compute_all_metrics(preds, refs, include_meteor=False, include_cider=False)
    meta = RunMeta(
        model_id="test-model",
        decode_strategy="greedy",
        weights_path="nowhere",
        tokenizer_dir="nowhere",
        n_samples=len(preds),
        max_length=40,
    )

    out_dir = write_run_artifacts(
        tmp_path / "runX",
        metrics=report,
        meta=meta,
        images=images,
        predictions=preds,
        references=refs,
        diagnostics=diags,
    )

    assert (out_dir / "metrics.json").is_file()
    assert (out_dir / "run_meta.json").is_file()
    assert (out_dir / "predictions.jsonl").is_file()
    assert (out_dir / "diagnostics.jsonl").is_file()
    assert (out_dir / "report.md").is_file()

    predictions_lines = (out_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(predictions_lines) == 2

    metadata = json.loads((out_dir / "run_meta.json").read_text(encoding="utf-8"))
    assert metadata["model_id"] == "test-model"
    assert metadata["n_samples"] == 2


_ = MIN_SAMPLES_FOR_CIDER  # — exposed re-export, exercised by import
