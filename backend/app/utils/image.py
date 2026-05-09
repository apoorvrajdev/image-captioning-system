"""Image-decoding utilities for the HTTP boundary.

The ML package's ``inference/image_loader.py`` reads from disk; the API
receives bytes in memory from a multipart upload. This module bridges the
two: it decodes raw bytes and runs them through the *same*
``preprocess_image_tensor`` the training pipeline uses, so train/serve
parity is preserved by construction.

TensorFlow imports are deferred until first call to keep app import cheap
(e.g. when running ``ruff`` or constructing the app for tests with stub
predictors).
"""

from __future__ import annotations

from typing import Any

ALLOWED_CONTENT_TYPES: frozenset[str] = frozenset(
    {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
        "image/bmp",
    }
)


class ImageDecodeError(ValueError):
    """Raised when uploaded bytes are not a recognisable image."""


def bytes_to_tensor(image_bytes: bytes) -> Any:
    """Decode an in-memory image into a model-ready tensor.

    Args:
        image_bytes: Raw bytes from a multipart upload (JPEG/PNG/WebP/BMP).

    Returns:
        ``tf.Tensor`` of shape ``[299, 299, 3]``, dtype ``float32``, with
        the InceptionV3 normalisation applied — i.e. exactly what
        ``CaptionPredictor.predict_tensor`` expects.

    Raises:
        ImageDecodeError: If the bytes can't be decoded as an image.
    """
    import tensorflow as tf

    from captioning.preprocessing.image import preprocess_image_tensor

    try:
        decoded = tf.io.decode_image(
            image_bytes,
            channels=3,
            expand_animations=False,
        )
    except (tf.errors.InvalidArgumentError, tf.errors.UnknownError) as exc:
        raise ImageDecodeError(f"Could not decode image bytes: {exc}") from exc

    return preprocess_image_tensor(decoded)
