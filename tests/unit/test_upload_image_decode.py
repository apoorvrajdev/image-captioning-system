"""Tests for ``app.utils.image.bytes_to_tensor`` with TensorFlow.

The dimension guard must run before ``tf.io.decode_image``, which allocates
the full image from its header, and an image within the limits must come out
exactly as the decode-then-preprocess path produced it before the guard.

TF-dependent; auto-skipped if TF is unavailable.
"""

from __future__ import annotations

import pytest

tf = pytest.importorskip("tensorflow")

from app.tests.image_bytes import encode, png_header  # noqa: E402
from app.utils.image import (  # noqa: E402
    ImageDecodeError,
    ImageTooLargeError,
    bytes_to_tensor,
)
from captioning.preprocessing.image import (  # noqa: E402
    INCEPTION_INPUT_SIZE,
    preprocess_image_tensor,
)


def test_oversized_image_never_reaches_the_decoder(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bytes] = []

    def recording_decoder(contents: bytes, **kwargs: object) -> None:
        calls.append(contents)
        raise tf.errors.InvalidArgumentError(None, None, "recorded, not decoded")

    monkeypatch.setattr(tf.io, "decode_image", recording_decoder)

    with pytest.raises(ImageTooLargeError):
        bytes_to_tensor(png_header(10_000, 5_000))
    assert calls == []

    # The recorder is live: an image within the limits reaches it.
    small = encode("PNG")
    with pytest.raises(ImageDecodeError):
        bytes_to_tensor(small)
    assert calls == [small]


@pytest.mark.parametrize(
    ("fmt", "mode"),
    [
        ("JPEG", "RGB"),
        ("JPEG", "L"),
        ("PNG", "RGB"),
        ("PNG", "RGBA"),
        ("PNG", "L"),
        ("BMP", "RGB"),
    ],
)
def test_image_within_the_limits_matches_the_unguarded_pipeline(fmt: str, mode: str) -> None:
    data = encode(fmt, size=(64, 48), mode=mode)

    out = bytes_to_tensor(data)

    # The path before the guard: decode to RGB, then the training preprocessing.
    expected = preprocess_image_tensor(
        tf.io.decode_image(data, channels=3, expand_animations=False)
    )
    assert tuple(out.shape) == (INCEPTION_INPUT_SIZE, INCEPTION_INPUT_SIZE, 3)
    assert out.dtype == tf.float32
    assert bool(tf.reduce_all(tf.equal(out, expected)))
    assert float(tf.reduce_min(out)) >= -1.0
    assert float(tf.reduce_max(out)) <= 1.0
