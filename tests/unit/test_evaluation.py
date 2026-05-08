"""Smoke tests for the BLEU evaluator.

We don't validate sacrebleu's correctness here — that's its own test suite.
We *do* validate our adapter: parallel-list shape handling, ragged references,
and that perfect predictions score 100.
"""

from __future__ import annotations

import pytest

sacrebleu = pytest.importorskip("sacrebleu")

from captioning.evaluation.bleu import corpus_bleu_score  # noqa: E402


def test_perfect_predictions_score_100() -> None:
    refs = [["a man riding a bike"], ["a dog in the park"]]
    preds = ["a man riding a bike", "a dog in the park"]
    assert corpus_bleu_score(preds, refs) == pytest.approx(100.0)


def test_completely_wrong_predictions_score_low() -> None:
    refs = [["a man riding a bike"], ["a dog in the park"]]
    preds = ["xyz qrs", "abc def"]
    score = corpus_bleu_score(preds, refs)
    assert 0.0 <= score < 5.0


def test_ragged_references_supported() -> None:
    refs = [
        ["a man riding a bike", "a person on a bicycle", "someone biking"],
        ["a dog in the park"],
    ]
    preds = ["a man riding a bike", "a dog in the park"]
    score = corpus_bleu_score(preds, refs)
    assert score > 50.0


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        corpus_bleu_score(["a", "b"], [["a"]])
