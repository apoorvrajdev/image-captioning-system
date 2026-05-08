"""Tests for ``captioning.preprocessing.image``.

TF-dependent; auto-skipped if TF is unavailable.
"""

from __future__ import annotations

import pytest

tf = pytest.importorskip("tensorflow")

from captioning.preprocessing.image import (  # noqa: E402
    INCEPTION_INPUT_SIZE,
    preprocess_image_tensor,
)


def test_output_shape() -> None:
    img = tf.random.uniform((480, 640, 3), minval=0, maxval=255, dtype=tf.int32)
    img = tf.cast(img, tf.uint8)
    out = preprocess_image_tensor(img)
    assert tuple(out.shape) == (INCEPTION_INPUT_SIZE, INCEPTION_INPUT_SIZE, 3)


def test_output_in_inception_range() -> None:
    """``inception_v3.preprocess_input`` maps [0, 255] → [-1, 1]."""
    img = tf.cast(
        tf.random.uniform((300, 300, 3), 0, 255, dtype=tf.int32),
        tf.uint8,
    )
    out = preprocess_image_tensor(img)
    assert float(tf.reduce_min(out)) >= -1.0 - 1e-6
    assert float(tf.reduce_max(out)) <= 1.0 + 1e-6


def test_deterministic_on_same_input() -> None:
    img = tf.cast(
        tf.random.uniform((400, 500, 3), 0, 255, dtype=tf.int32),
        tf.uint8,
    )
    a = preprocess_image_tensor(img)
    b = preprocess_image_tensor(img)
    assert tf.reduce_all(tf.equal(a, b))
