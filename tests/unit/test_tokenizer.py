"""Tests for ``captioning.preprocessing.tokenizer.CaptionTokenizer``.

These are TF-dependent and slow to import; pytest auto-skips if TF is missing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

tf = pytest.importorskip("tensorflow")

from captioning.preprocessing.tokenizer import (  # noqa: E402
    VOCAB_JSON_FILENAME,
    VOCAB_PICKLE_FILENAME,
    CaptionTokenizer,
)


def test_fit_then_encode_decode_roundtrip(tiny_caption_corpus: list[str]) -> None:
    tok = CaptionTokenizer(vocab_size=200, max_length=20)
    tok.fit(tiny_caption_corpus)

    ids = tok.encode([tiny_caption_corpus[0]])
    assert ids.shape == (1, 20)

    # Decoding the first non-padding id should produce a known token.
    first_id = int(ids[0, 0].numpy())
    word = tok.decode_id(first_id)
    assert isinstance(word, str)


def test_save_load_round_trip_matches_original(
    tiny_caption_corpus: list[str], tmp_artifacts_dir: Path
) -> None:
    tok = CaptionTokenizer(vocab_size=200, max_length=20)
    tok.fit(tiny_caption_corpus)
    tok.save(tmp_artifacts_dir)

    assert (tmp_artifacts_dir / VOCAB_PICKLE_FILENAME).is_file()
    assert (tmp_artifacts_dir / VOCAB_JSON_FILENAME).is_file()

    loaded = CaptionTokenizer.load(tmp_artifacts_dir, vocab_size=200, max_length=20)
    assert loaded.vocabulary == tok.vocabulary
    # Encoding should match exactly
    ids_a = tok.encode([tiny_caption_corpus[0]]).numpy().tolist()
    ids_b = loaded.encode([tiny_caption_corpus[0]]).numpy().tolist()
    assert ids_a == ids_b


def test_unfitted_tokenizer_raises(tmp_artifacts_dir: Path) -> None:
    tok = CaptionTokenizer(vocab_size=200, max_length=20)
    with pytest.raises(RuntimeError, match="not fitted"):
        _ = tok.vocabulary
    with pytest.raises(RuntimeError, match="not fitted"):
        tok.encode(["hello"])
    with pytest.raises(RuntimeError, match="not fitted"):
        tok.save(tmp_artifacts_dir)


def test_max_length_is_respected(tiny_caption_corpus: list[str]) -> None:
    tok = CaptionTokenizer(vocab_size=200, max_length=10)
    tok.fit(tiny_caption_corpus)
    long_caption = " ".join(["[start]"] + ["word"] * 30 + ["[end]"])
    ids = tok.encode([long_caption])
    assert ids.shape == (1, 10)
