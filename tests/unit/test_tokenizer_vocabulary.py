"""Tests for ``read_vocabulary``: JSON first, a restricted pickle fallback, validation.

TensorFlow-free: ``CaptionTokenizer.load`` calls ``read_vocabulary`` before it
imports TensorFlow.
"""

from __future__ import annotations

import json
import operator
import pickle
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from captioning.preprocessing.tokenizer import (
    VOCAB_JSON_FILENAME,
    VOCAB_PICKLE_FILENAME,
    read_vocabulary,
)

VOCAB = ["", "[UNK]", "a", "[start]", "[end]", "dog"]


def _write_json(directory: Path, vocab: object) -> None:
    (directory / VOCAB_JSON_FILENAME).write_text(json.dumps(vocab), encoding="utf-8")


def _write_pickle(directory: Path, vocab: object) -> None:
    (directory / VOCAB_PICKLE_FILENAME).write_bytes(pickle.dumps(vocab))


class _CallsAGlobal:
    """Pickles as a call to ``operator.truediv(1, 0)``: harmless if it ever runs."""

    def __reduce__(self) -> tuple[Callable[[int, int], float], tuple[int, int]]:
        return (operator.truediv, (1, 0))


def test_json_vocabulary_is_read_in_id_order(tmp_path: Path) -> None:
    _write_json(tmp_path, VOCAB)

    assert read_vocabulary(tmp_path, vocab_size=100) == VOCAB


def test_json_wins_and_the_pickle_is_never_read(tmp_path: Path) -> None:
    _write_json(tmp_path, VOCAB)
    (tmp_path / VOCAB_PICKLE_FILENAME).write_bytes(b"not a pickle")

    assert read_vocabulary(tmp_path, vocab_size=100) == VOCAB


def test_plain_pickle_is_read_when_there_is_no_json(tmp_path: Path) -> None:
    _write_pickle(tmp_path, VOCAB)

    assert read_vocabulary(tmp_path, vocab_size=100) == VOCAB


def test_numpy_string_pickle_is_read_as_plain_strings(tmp_path: Path) -> None:
    """The notebook's pickles hold ``numpy.str_`` entries, like Hub ``v2.0.0``'s ``vocab.pkl``."""
    _write_pickle(tmp_path, [np.str_(token) for token in VOCAB])

    tokens = read_vocabulary(tmp_path, vocab_size=100)

    assert tokens == VOCAB
    assert {type(token) for token in tokens} == {str}


def test_pickle_naming_any_other_global_is_refused_before_it_runs(tmp_path: Path) -> None:
    _write_pickle(tmp_path, [*VOCAB, _CallsAGlobal()])

    with pytest.raises(ValueError, match=r"refused global _?operator\.truediv"):
        read_vocabulary(tmp_path, vocab_size=100)


@pytest.mark.parametrize("write", [_write_json, _write_pickle], ids=["json", "pickle"])
@pytest.mark.parametrize(
    ("vocab", "message"),
    [
        ({"a": 1}, "must be a list of strings, got dict"),
        (["", "[UNK]", 5], "has a non-string entry at id 2: int"),
        (
            ["a", "dog"],
            "must start with the padding and OOV tokens ['', '[UNK]'], got ['a', 'dog']",
        ),
        (["[UNK]", "", "a"], "must start with the padding and OOV tokens"),
        ([], "must start with the padding and OOV tokens ['', '[UNK]'], got []"),
        (["", "[UNK]", "a", "dog", "a"], "repeats the token 'a'"),
    ],
)
def test_inconsistent_vocabularies_are_rejected(
    tmp_path: Path, write: Callable[[Path, object], None], vocab: object, message: str
) -> None:
    write(tmp_path, vocab)

    with pytest.raises(ValueError, match=re.escape(message)):
        read_vocabulary(tmp_path, vocab_size=100)


def test_vocabulary_larger_than_the_model_is_rejected(tmp_path: Path) -> None:
    _write_json(tmp_path, VOCAB)

    with pytest.raises(ValueError, match=re.escape("has 6 entries, more than vocab_size=5")):
        read_vocabulary(tmp_path, vocab_size=5)


@pytest.mark.parametrize(
    "raw",
    [b'["", "[UNK]", "a"', b"\xff\xfe not utf-8", b""],
    ids=["truncated", "not-utf8", "empty"],
)
def test_unparseable_json_is_rejected_by_name(tmp_path: Path, raw: bytes) -> None:
    (tmp_path / VOCAB_JSON_FILENAME).write_bytes(raw)

    with pytest.raises(ValueError, match="Could not parse tokenizer vocabulary .*vocab.json"):
        read_vocabulary(tmp_path, vocab_size=100)


def test_truncated_pickle_is_rejected_by_name(tmp_path: Path) -> None:
    (tmp_path / VOCAB_PICKLE_FILENAME).write_bytes(pickle.dumps(VOCAB)[:10])

    with pytest.raises(ValueError, match="Could not read tokenizer vocabulary .*vocab.pkl"):
        read_vocabulary(tmp_path, vocab_size=100)


def test_missing_vocabulary_names_both_files(tmp_path: Path) -> None:
    with pytest.raises(
        FileNotFoundError, match="Expected 'vocab.json' \\(preferred\\) or 'vocab.pkl'"
    ):
        read_vocabulary(tmp_path, vocab_size=100)
