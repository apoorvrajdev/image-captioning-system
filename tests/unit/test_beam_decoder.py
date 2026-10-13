"""Beam-search decoder unit tests.

The full TF decoder forward path is exercised by the parity audit and the
smoke test in ``scripts/predict.py``. Here we test the *algorithmic* pieces
of beam search in isolation:

    * Length penalty correctly rescales scores.
    * Repetition penalty downweights seen tokens.
    * n-gram blocker forbids exact-repeat n-grams.
    * Detokeniser strips ``[start]`` / ``[end]`` and stops at ``[end]``.
    * Beam width 1 produces exactly the greedy caption.

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
from captioning.inference.greedy import generate_caption_greedy


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


# ---- Beam width 1 against greedy --------------------------------------------

_VOCAB = ["", "[UNK]", "[start]", "[end]", "a", "the", "dog", "cat", "on", "grass"]
_IDS = {word: i for i, word in enumerate(_VOCAB)}
_MAX_LENGTH = 8


class _WordTokenizer:
    """Whitespace tokenizer over ``_VOCAB``, with the calls both decoders make."""

    def encode(self, texts: list[str]) -> np.ndarray:
        ids = [_IDS[word] for word in texts[0].split()][:_MAX_LENGTH]
        return np.array([ids + [0] * (_MAX_LENGTH - len(ids))])

    def decode_id(self, idx) -> str:
        return _VOCAB[int(idx)]

    def word_to_id(self, word: str) -> int:
        return _IDS[word]


class _BigramModel:
    """Decoder whose next-token distribution depends on the token at each position.

    Both decoders read position ``i`` after feeding ``i + 1`` tokens, so the
    distribution follows the caption built so far, not just the step index.
    """

    def __init__(self, table: dict[str, dict[str, float]]) -> None:
        self.table = np.full((len(_VOCAB), len(_VOCAB)), 1e-4, dtype=np.float32)
        for prev, following in table.items():
            for word, prob in following.items():
                self.table[_IDS[prev], _IDS[word]] = prob
        self.table /= self.table.sum(axis=1, keepdims=True)
        self.cnn_model = lambda img: img
        self.encoder = lambda x, training: x

    def decoder(self, tokens, encoded, training, mask):
        import tensorflow as tf

        return tf.convert_to_tensor(self.table[np.asarray(tokens)])


_CLEAR_PATH = {
    "[start]": {"a": 0.9},
    "a": {"dog": 0.8},
    "dog": {"on": 0.7},
    "on": {"grass": 0.8},
    "grass": {"[end]": 0.9},
}
# [end] is the runner-up after "dog", and "a dog" outscores "a dog on grass" at
# every length penalty. A search that kept finished candidates from outside the
# beam would return the shorter caption; at width 1 it must be discarded. The
# rows sum to about 1, so normalisation keeps these values.
_END_AS_RUNNER_UP = {
    "[start]": {"a": 0.9, "the": 0.0995},
    "a": {"dog": 0.9, "cat": 0.0995},
    "dog": {"on": 0.5, "[end]": 0.45, "cat": 0.0495},
    "on": {"grass": 0.5, "cat": 0.4995},
    "grass": {"[end]": 0.5, "on": 0.4995},
}
# No [end] is ever likely, so both decoders run out of budget.
_NO_END = {"[start]": {"a": 0.9}, "a": {"dog": 0.9}, "dog": {"dog": 0.9}}


@pytest.mark.parametrize("length_penalty", [0.0, 0.7, 1.0])
@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (_CLEAR_PATH, "a dog on grass"),
        (_END_AS_RUNNER_UP, "a dog on grass"),
        (_NO_END, "a" + " dog" * (_MAX_LENGTH - 2)),
    ],
    ids=["clear-path", "end-as-runner-up", "no-end"],
)
def test_beam_width_one_matches_greedy(
    table: dict[str, dict[str, float]], expected: str, length_penalty: float
) -> None:
    import tensorflow as tf

    model, tokenizer = _BigramModel(table), _WordTokenizer()
    image = tf.zeros((299, 299, 3), dtype=tf.float32)

    greedy = generate_caption_greedy(model, tokenizer, image, _MAX_LENGTH)
    beam = generate_caption_beam(
        model, tokenizer, image, _MAX_LENGTH, beam_width=1, length_penalty=length_penalty
    )

    assert greedy == expected
    assert beam == greedy


def test_a_wider_beam_can_beat_greedy_on_the_same_model() -> None:
    """Shows the fixture can tell the decoders apart, so width 1 matching greedy means something."""
    import tensorflow as tf

    garden_path = {
        "[start]": {"a": 0.55, "the": 0.45},
        "a": {"dog": 0.35, "cat": 0.3, "grass": 0.3},
        "the": {"dog": 0.95},
        "dog": {"[end]": 0.9},
    }
    model, tokenizer = _BigramModel(garden_path), _WordTokenizer()
    image = tf.zeros((299, 299, 3), dtype=tf.float32)

    assert generate_caption_greedy(model, tokenizer, image, _MAX_LENGTH) == "a dog"
    assert (
        generate_caption_beam(
            model, tokenizer, image, _MAX_LENGTH, beam_width=2, length_penalty=0.0
        )
        == "the dog"
    )
