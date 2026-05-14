"""Tests for the opt-in training-stability primitives.

Covers:
    * ``label_smoothed_crossentropy`` returns a per-token loss tensor with the
      same shape as the baseline sparse loss, and reduces to it at smoothing=0.
    * ``WarmupCosineDecay`` produces the expected piecewise schedule.
    * ``build_loss`` / ``build_learning_rate`` dispatch correctly on config.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from captioning.training.losses import build_loss, label_smoothed_crossentropy
from captioning.training.schedules import WarmupCosineDecay, build_learning_rate

# ---- Label smoothing -------------------------------------------------------


def test_label_smoothed_loss_returns_per_token_shape() -> None:
    import tensorflow as tf

    vocab = 5
    loss_fn = label_smoothed_crossentropy(0.1, vocab)
    y_true = tf.constant([[1, 2, 0]], dtype=tf.int32)
    y_pred = tf.constant(
        [
            [
                [0.05, 0.85, 0.05, 0.025, 0.025],
                [0.05, 0.05, 0.85, 0.025, 0.025],
                [0.85, 0.05, 0.05, 0.025, 0.025],
            ]
        ],
        dtype=tf.float32,
    )
    loss = loss_fn(y_true, y_pred).numpy()
    assert loss.shape == (1, 3)
    # The first two tokens are confidently correct → low loss.
    assert loss[0, 0] < 1.0
    assert loss[0, 1] < 1.0


def test_label_smoothing_with_zero_returns_baseline_loss() -> None:
    loss = build_loss(0.0, vocab_size=10)
    # The baseline SparseCategoricalCrossentropy is an instance, not a function.
    import tensorflow as tf

    assert isinstance(loss, tf.keras.losses.SparseCategoricalCrossentropy)


def test_label_smoothing_is_higher_than_unsmoothed_on_perfect_prediction() -> None:
    """Smoothing punishes overconfidence — perfect one-hot prediction gets a
    higher per-token loss with smoothing > 0 than without."""
    import tensorflow as tf

    vocab = 5
    y_true = tf.constant([[1]], dtype=tf.int32)
    one_hot_pred = tf.constant([[[0.0, 1.0, 0.0, 0.0, 0.0]]], dtype=tf.float32)

    smoothed = label_smoothed_crossentropy(0.1, vocab)(y_true, one_hot_pred).numpy()
    unsmoothed = -np.log(1.0)  # sparse cross-entropy on argmax==y_true is 0
    assert smoothed[0, 0] > unsmoothed + 1e-3


# ---- Learning-rate schedule -----------------------------------------------


def test_warmup_cosine_zero_at_step_zero() -> None:
    import tensorflow as tf

    schedule = WarmupCosineDecay(peak_learning_rate=1.0, warmup_steps=10, decay_steps=100)
    assert float(schedule(tf.constant(0, dtype=tf.int64))) == pytest.approx(0.0)


def test_warmup_cosine_peaks_at_end_of_warmup() -> None:
    import tensorflow as tf

    schedule = WarmupCosineDecay(peak_learning_rate=1.0, warmup_steps=10, decay_steps=100)
    assert float(schedule(tf.constant(10, dtype=tf.int64))) == pytest.approx(1.0, abs=1e-3)


def test_warmup_cosine_floors_at_end_of_decay() -> None:
    import tensorflow as tf

    schedule = WarmupCosineDecay(
        peak_learning_rate=1.0,
        warmup_steps=10,
        decay_steps=100,
        min_learning_rate=0.1,
    )
    final = float(schedule(tf.constant(110, dtype=tf.int64)))
    assert final == pytest.approx(0.1, abs=1e-3)


def test_warmup_cosine_is_monotone_during_warmup() -> None:
    import tensorflow as tf

    schedule = WarmupCosineDecay(peak_learning_rate=1.0, warmup_steps=10, decay_steps=100)
    values = [float(schedule(tf.constant(s, dtype=tf.int64))) for s in range(11)]
    assert all(b >= a for a, b in pairwise(values))


def test_build_learning_rate_returns_float_for_constant() -> None:
    lr = build_learning_rate(
        schedule="constant",
        peak_learning_rate=1e-3,
        warmup_steps=0,
        decay_steps=10,
        min_learning_rate=0.0,
    )
    assert lr == 1e-3


def test_build_learning_rate_returns_schedule_for_cosine() -> None:
    lr = build_learning_rate(
        schedule="cosine",
        peak_learning_rate=1e-3,
        warmup_steps=5,
        decay_steps=50,
        min_learning_rate=0.0,
    )
    assert isinstance(lr, WarmupCosineDecay)


def test_build_learning_rate_rejects_unknown_schedule() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        build_learning_rate(
            schedule="square_wave",
            peak_learning_rate=1.0,
            warmup_steps=0,
            decay_steps=10,
            min_learning_rate=0.0,
        )
