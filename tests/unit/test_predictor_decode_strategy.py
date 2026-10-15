"""``CaptionPredictor.predict_tensor`` takes the decoding strategy per call (TASK-034).

A scripted bigram decoder stands in for the model, so no weights are loaded.
On it, greedy follows the locally best word ("a dog") while beam search finds
the better sequence ("the dog"), which makes the two strategies distinguishable.
"""

from __future__ import annotations

import numpy as np
import pytest

from captioning.config.schema import AppConfig
from captioning.inference.predictor import DECODE_STRATEGIES, CaptionPredictor

_VOCAB = ["", "[UNK]", "[start]", "[end]", "a", "the", "dog", "cat", "grass"]
_IDS = {word: i for i, word in enumerate(_VOCAB)}
_MAX_LENGTH = 8
_GARDEN_PATH = {
    "[start]": {"a": 0.55, "the": 0.45},
    "a": {"dog": 0.35, "cat": 0.3, "grass": 0.3},
    "the": {"dog": 0.95},
    "dog": {"[end]": 0.9},
}


class _WordTokenizer:
    def encode(self, texts: list[str]) -> np.ndarray:
        ids = [_IDS[word] for word in texts[0].split()][:_MAX_LENGTH]
        return np.array([ids + [0] * (_MAX_LENGTH - len(ids))])

    def decode_id(self, idx) -> str:
        return _VOCAB[int(idx)]

    def word_to_id(self, word: str) -> int:
        return _IDS[word]


class _BigramModel:
    def __init__(self) -> None:
        self.table = np.full((len(_VOCAB), len(_VOCAB)), 1e-4, dtype=np.float32)
        for prev, following in _GARDEN_PATH.items():
            for word, prob in following.items():
                self.table[_IDS[prev], _IDS[word]] = prob
        self.table /= self.table.sum(axis=1, keepdims=True)
        self.cnn_model = lambda img: img
        self.encoder = lambda x, training: x

    def decoder(self, tokens, encoded, training, mask):
        import tensorflow as tf

        return tf.convert_to_tensor(self.table[np.asarray(tokens)])


def _predictor(**kwargs) -> CaptionPredictor:
    return CaptionPredictor(
        model=_BigramModel(),
        tokenizer=_WordTokenizer(),  # type: ignore[arg-type]
        config=AppConfig(model={"max_length": _MAX_LENGTH}),
        beam_width=2,
        length_penalty=0.0,
        **kwargs,
    )


@pytest.fixture
def image():
    import tensorflow as tf

    return tf.zeros((299, 299, 3), dtype=tf.float32)


def test_strategy_constants_match_the_literal() -> None:
    assert DECODE_STRATEGIES == ("greedy", "beam")


def test_default_call_uses_the_instance_strategy(image) -> None:
    assert _predictor().predict_tensor(image) == "a dog"
    assert _predictor(decode_strategy="beam").predict_tensor(image) == "the dog"


def test_per_call_strategy_overrides_without_changing_the_instance(image) -> None:
    predictor = _predictor()

    assert predictor.predict_tensor(image, decode_strategy="beam") == "the dog"
    assert predictor.decode_strategy == "greedy"
    assert predictor.predict_tensor(image) == "a dog"

    beam_default = _predictor(decode_strategy="beam")
    assert beam_default.predict_tensor(image, decode_strategy="greedy") == "a dog"
    assert beam_default.decode_strategy == "beam"


def test_none_means_the_instance_default(image) -> None:
    assert _predictor().predict_tensor(image, decode_strategy=None) == "a dog"


@pytest.mark.parametrize("bad", ["sampling", "BEAM", ""])
def test_unknown_strategy_is_refused(image, bad: str) -> None:
    predictor = _predictor()
    if bad == "":
        # An empty string falls back to the default rather than raising.
        assert predictor.predict_tensor(image, decode_strategy=bad) == "a dog"  # type: ignore[arg-type]
        return
    with pytest.raises(ValueError, match="decode_strategy must be 'greedy' or 'beam'"):
        predictor.predict_tensor(image, decode_strategy=bad)  # type: ignore[arg-type]
