"""Image-decoding utilities for the HTTP boundary.

The ML package's ``inference/image_loader.py`` reads from disk; the API
receives bytes in memory from a multipart upload. This module bridges the
two: it decodes raw bytes and runs them through the *same*
``preprocess_image_tensor`` the training pipeline uses, so train/serve
parity is preserved by construction.

Before anything decodes, ``check_image_dimensions`` reads the image header
and rejects an image too large to decode safely: the compressed upload cap
doesn't bound the decoded size.

TensorFlow imports are deferred until first call to keep app import cheap
(e.g. when running ``ruff`` or constructing the app for tests with stub
predictors).
"""

from __future__ import annotations

import io
from typing import Any

from PIL import Image

# WebP isn't listed: TF 2.15's ``decode_image`` can't decode it, and serving
# must decode through the training path, so a WebP upload could never succeed.
ALLOWED_CONTENT_TYPES: frozenset[str] = frozenset(
    {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/bmp",
    }
)

# Upload limits on decoded size, checked against the header before any decode.
# The model only sees 299x299, so they bound memory, not caption quality.
# Decoding allocates 3 bytes per pixel (uint8 RGB) and the Resizing layer then
# makes a float32 copy (12 bytes per pixel), so a 40 MP image needs roughly
# 600 MB per request. That figure is derived from the dtypes, not measured.
# 40 MP admits 12-36 MP camera and phone photos; the 16,384 px side admits
# panoramas while bounding per-row decoder buffers.
MAX_IMAGE_PIXELS = 40_000_000
MAX_IMAGE_SIDE = 16_384

# Formats whose header is read. TF 2.15's decode_image also takes GIF, but it
# sizes a GIF by its largest frame, which the header read doesn't see, so GIF
# bytes are rejected (GIF isn't on the allow-list). WebP is never parsed:
# Pillow's WebP open allocates the full canvas, and TF 2.15 can't decode WebP.
_HEADER_FORMATS = ("JPEG", "PNG", "BMP")


class ImageDecodeError(ValueError):
    """Raised when uploaded bytes are not a recognisable image."""


class ImageTooLargeError(ImageDecodeError):
    """Raised when an image's dimensions exceed the upload limits.

    Subclasses ``ImageDecodeError`` so a handler that only catches that class
    still answers with a client error, never a 500.
    """


def check_image_dimensions(
    image_bytes: bytes,
    *,
    max_side: int = MAX_IMAGE_SIDE,
    max_pixels: int = MAX_IMAGE_PIXELS,
) -> tuple[int, int]:
    """Read the image header and reject an image too large to decode safely.

    ``tf.io.decode_image`` allocates the full-resolution tensor from the
    header's dimensions before it reads any pixel data, so a small file can
    claim a huge image. ``Image.open`` parses only the header here; no pixel
    is decoded.

    Args:
        image_bytes: Raw upload bytes.
        max_side: Largest width or height accepted, in pixels.
        max_pixels: Largest ``width * height`` accepted.

    Returns:
        ``(width, height)`` read from the header.

    Raises:
        ImageTooLargeError: If either side or the pixel count is over its limit.
        ImageDecodeError: If the bytes aren't a JPEG, PNG or BMP with a
            readable header and positive dimensions.
    """
    too_large = (
        f"Image dimensions exceed the limit of {max_side} pixels per side "
        f"and {max_pixels:,} pixels in total."
    )
    try:
        with Image.open(io.BytesIO(image_bytes), formats=_HEADER_FORMATS) as image:
            width, height = image.size
    except Image.DecompressionBombError as exc:
        # Pillow's own ceiling (about 179 MP) is far above max_pixels.
        raise ImageTooLargeError(too_large) from exc
    except Exception as exc:
        # Pillow's header parsers raise many exception types on hostile input.
        # Each one means the upload isn't an image this service accepts.
        raise ImageDecodeError(
            "Could not decode image bytes: expected a JPEG, PNG or BMP image."
        ) from exc

    if width <= 0 or height <= 0:
        raise ImageDecodeError("Could not decode image bytes: invalid image dimensions.")
    if width > max_side or height > max_side or width * height > max_pixels:
        raise ImageTooLargeError(too_large)
    return width, height


def bytes_to_tensor(image_bytes: bytes) -> Any:
    """Decode an in-memory image into a model-ready tensor.

    Args:
        image_bytes: Raw bytes from a multipart upload (JPEG/PNG/BMP).

    Returns:
        ``tf.Tensor`` of shape ``[299, 299, 3]``, dtype ``float32``, with
        the InceptionV3 normalisation applied — i.e. exactly what
        ``CaptionPredictor.predict_tensor`` expects.

    Raises:
        ImageTooLargeError: If the header dimensions exceed the limits.
        ImageDecodeError: If the bytes can't be decoded as an image.
    """
    # Runs before TensorFlow is even imported: the decode below allocates the
    # whole image as soon as it has read the header.
    check_image_dimensions(image_bytes)

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
