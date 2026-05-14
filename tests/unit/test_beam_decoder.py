"""Beam-search decoder unit tests.

The full TF decoder forward path is exercised by the parity audit and the
smoke test in ``scripts/predict.py``. Here we test the *algorithmic* pieces
of beam search in isolation:

    * Length penalty correctly rescales scores.
    * Repetition penalty downweights seen tokens.
    * n-gram blocker forbids exact-repeat n-grams.
    * Detokeniser strips ``[start]`` / ``[end]`` and stops at ``[end]``.

A small fake model is used to verify end-to-end search behaviour without
loading TensorFlow weights.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from captioning.inference.beam import (
    _apply_repetition_penalty,
    _Beam,
    _blocks_repeat_ngram,
    _detokenize,
    _length_normalised,
    generate_caption_beam,
)


def test_length_penalty_zero_returns_raw_score() -> None:
    b = _Beam(token_ids=[1, 2, 3], score=-5.0)
    assert _length_normalised(b, 0.0) == -5.0


def test_length_penalty_one_divides_by_length() -> None:
    b = _Beam(token_ids=[1, 2, 3, 4], score=-6.0)  # length=3
    assert _length_normalised(b, 1.0) == pytest.approx(-2.0)


def test_repetition_penalty_downweights_seen_tokens() -> None:
    log_probs = np.array([-1.0, -2.0, -3.0, -4.0])
    out = _apply_repetition_penalty(log_probs.copy(), history_ids={1, 3}, penalty=2.0)
    # Penalty subtracts log(2) ~ 0.693 from seen-token log-probs.
    assert out[0] == pytest.approx(-1.0)
    assert out[1] == pytest.approx(-2.0 - np.log(2.0))
    assert out[2] == pytest.approx(-3.0)
    assert out[3] == pytest.approx(-4.0 - np.log(2.0))


def test_repetition_penalty_one_is_noop() -> None:
    log_probs = np.array([-1.0, -2.0, -3.0])
    out = _apply_repetition_penalty(log_probs.copy(), history_ids={0, 1}, penalty=1.0)
    np.testing.assert_array_equal(out, log_probs)


def test_blocks_repeat_ngram_detects_repeat() -> None:
    # seq ends with [4, 5]; appending 6 forms trigram [4, 5, 6] not present.
    assert not _blocks_repeat_ngram([1, 2, 3, 4, 5], 6, n=3)
    # Now seq contains [4, 5, 6]; appending 6 still wouldn't form a repeat.
    assert not _blocks_repeat_ngram([4, 5, 6, 4, 5], 7, n=3)
    # seq has [4, 5, 6] AND ends with [4, 5]; appending 6 repeats [4, 5, 6].
    assert _blocks_repeat_ngram([4, 5, 6, 1, 4, 5], 6, n=3)


def test_blocks_repeat_ngram_zero_size_disables() -> None:
    assert not _blocks_repeat_ngram([1, 1, 1], 1, n=0)


def test_detokenize_stops_at_end_and_skips_special_tokens() -> None:
    tokenizer = MagicMock()
    # ids: [start]=1, "a"=2, "man"=3, [end]=4
    table = {1: "[start]", 2: "a", 3: "man", 4: "[end]"}
    tokenizer.decode_id = lambda i: table[i]
    out = _detokenize([1, 2, 3, 4, 99], tokenizer, end_id=4)
    assert out == "a man"


# ---- End-to-end beam search with a fake model -----------------------------


class _FakeModel:
    """Decoder fixture that always assigns the highest probability to ``best_id``.

    The decoder output is the only piece beam search cares about; we stub the
    CNN / encoder to identity-like behaviour so the whole inference pass runs
    without TF being loaded.
    """

    def __init__(self, vocab_size: int, best_id: int) -> None:
        self.vocab_size = vocab_size
        self.best_id = best_id

        self.cnn_model = MagicMock(side_effect=self._identity_image)
        self.encoder = MagicMock(side_effect=self._identity_encoder)
        self.decoder = MagicMock(side_effect=self._decoder_step)

    def _identity_image(self, img):
        return img

    def _identity_encoder(self, x, training):
        return x

    def _decoder_step(self, tokens, encoded, training, mask):
        import tensorflow as tf

        batch = int(tf.shape(tokens)[0])
        seq_len = int(tf.shape(tokens)[1])
        probs = np.full((batch, seq_len, self.vocab_size), 1e-3, dtype=np.float32)
        probs[:, :, self.best_id] = 0.999
        # Normalise so each row over vocab sums to ~1.
        probs /= probs.sum(axis=-1, keepdims=True)
        return tf.convert_to_tensor(probs)


def test_beam_search_emits_caption_when_model_prefers_end_token() -> None:
    import tensorflow as tf

    tokenizer = MagicMock()
    # vocab: 0=pad, 1=[start], 2=[end], 3="dog"
    word_to_id_table = {"[start]": 1, "[end]": 2}
    decode_id_table = {0: "", 1: "[start]", 2: "[end]", 3: "dog"}
    tokenizer.word_to_id = lambda w: word_to_id_table[w]
    tokenizer.decode_id = lambda i: decode_id_table[i]

    # Model always predicts "dog" (id=3).
    model = _FakeModel(vocab_size=4, best_id=3)
    image = tf.zeros((299, 299, 3), dtype=tf.float32)

    caption = generate_caption_beam(
        model,
        tokenizer,
        image,
        max_length=6,
        beam_width=2,
        length_penalty=0.0,
    )
    # With no length penalty and no repetition penalty, the greedy-ish path
    # outputs repeated "dog" until max_length. We just assert it produced
    # *something* and didn't crash.
    assert caption.startswith("dog")


def test_beam_search_terminates_on_eos() -> None:
    """Beam search must produce a clean caption when the model emits [end]."""
    import tensorflow as tf

    tokenizer = MagicMock()
    word_to_id_table = {"[start]": 1, "[end]": 2}
    decode_id_table = {0: "", 1: "[start]", 2: "[end]", 3: "dog"}
    tokenizer.word_to_id = lambda w: word_to_id_table[w]
    tokenizer.decode_id = lambda i: decode_id_table[i]

    # Step 0: prefer "dog"; step 1+: prefer [end].
    class _EosFakeModel(_FakeModel):
        def _decoder_step(self, tokens, encoded, training, mask):
            batch = int(tf.shape(tokens)[0])
            seq_len = int(tf.shape(tokens)[1])
            probs = np.full((batch, seq_len, self.vocab_size), 1e-3, dtype=np.float32)
            probs[:, 0, 3] = 0.99  # at position 0 prefer "dog"
            for pos in range(1, seq_len):
                probs[:, pos, 2] = 0.99  # afterwards prefer [end]
            probs /= probs.sum(axis=-1, keepdims=True)
            return tf.convert_to_tensor(probs)

    model = _EosFakeModel(vocab_size=4, best_id=3)
    caption = generate_caption_beam(
        model,
        tokenizer,
        tf.zeros((299, 299, 3), dtype=tf.float32),
        max_length=6,
        beam_width=2,
    )
    assert caption == "dog"
