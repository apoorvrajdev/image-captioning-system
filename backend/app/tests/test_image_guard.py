"""Tests for the upload dimension guard in ``app.utils.image``.

``check_image_dimensions`` reads the image header and must reject an image
over the limits before ``bytes_to_tensor`` decodes it. Oversized inputs are
header-only bytes (``image_bytes``), so nothing large is allocated, and
TensorFlow is never imported here.
"""

from __future__ import annotations

import struct
import sys

import pytest

from app.tests.image_bytes import encode, jpeg_header, png_header
from app.utils import image as image_module
from app.utils.image import (
    MAX_IMAGE_PIXELS,
    MAX_IMAGE_SIDE,
    ImageDecodeError,
    ImageTooLargeError,
    bytes_to_tensor,
    check_image_dimensions,
)

LIMIT_MESSAGE = (
    "Image dimensions exceed the limit of 16384 pixels per side and 40,000,000 pixels in total."
)


def _bmp_with_width(width: int) -> bytes:
    data = bytearray(encode("BMP"))
    data[18:22] = struct.pack("<i", width)
    return bytes(data)


def _png_with_bad_ihdr_crc() -> bytes:
    data = bytearray(png_header(32, 24))
    data[29] ^= 0xFF  # first byte of the IHDR checksum
    return bytes(data)


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "BMP"])
def test_accepts_a_small_image(fmt: str) -> None:
    assert check_image_dimensions(encode(fmt, size=(32, 24))) == (32, 24)


@pytest.mark.parametrize("header", [png_header, jpeg_header])
def test_accepts_dimensions_exactly_at_the_limits(header) -> None:
    assert MAX_IMAGE_PIXELS == 8_000 * 5_000
    assert check_image_dimensions(header(8_000, 5_000)) == (8_000, 5_000)
    assert check_image_dimensions(header(MAX_IMAGE_SIDE, 1)) == (MAX_IMAGE_SIDE, 1)
    assert check_image_dimensions(header(1, MAX_IMAGE_SIDE)) == (1, MAX_IMAGE_SIDE)


@pytest.mark.parametrize("header", [png_header, jpeg_header])
def test_rejects_a_pixel_count_over_the_limit(header) -> None:
    # Both sides are within MAX_IMAGE_SIDE; only the product is over.
    with pytest.raises(ImageTooLargeError):
        check_image_dimensions(header(8_000, 5_001))


@pytest.mark.parametrize("header", [png_header, jpeg_header])
@pytest.mark.parametrize("size", [(MAX_IMAGE_SIDE + 1, 1), (1, MAX_IMAGE_SIDE + 1)])
def test_rejects_a_side_over_the_limit(header, size: tuple[int, int]) -> None:
    with pytest.raises(ImageTooLargeError):
        check_image_dimensions(header(*size))


def test_rejects_dimensions_past_pillows_own_bomb_ceiling() -> None:
    # 65535 x 65535 makes Image.open itself raise DecompressionBombError.
    with pytest.raises(ImageTooLargeError):
        check_image_dimensions(png_header(65_535, 65_535))


def test_limit_message_is_concise_and_names_only_the_limits() -> None:
    with pytest.raises(ImageTooLargeError) as info:
        check_image_dimensions(png_header(10_000, 5_000))
    assert str(info.value) == LIMIT_MESSAGE


def test_honours_explicit_limits() -> None:
    small = encode("PNG", size=(32, 24))
    with pytest.raises(ImageTooLargeError):
        check_image_dimensions(small, max_side=31)
    with pytest.raises(ImageTooLargeError):
        check_image_dimensions(small, max_pixels=32 * 24 - 1)
    assert check_image_dimensions(small, max_side=32, max_pixels=32 * 24) == (32, 24)


@pytest.mark.parametrize(
    ("data", "error"),
    [
        pytest.param(b"", ImageDecodeError, id="empty"),
        pytest.param(b"not an image at all", ImageDecodeError, id="garbage"),
        pytest.param(b"\xff\xd8\xff", ImageDecodeError, id="jpeg-soi-only"),
        pytest.param(jpeg_header(32, 24)[:10], ImageDecodeError, id="jpeg-cut-in-sof"),
        pytest.param(png_header(32, 24)[:20], ImageDecodeError, id="png-cut-in-ihdr"),
        pytest.param(png_header(32, 24)[:33], ImageDecodeError, id="png-ends-after-ihdr"),
        pytest.param(_png_with_bad_ihdr_crc(), ImageDecodeError, id="png-bad-ihdr-crc"),
        pytest.param(png_header(0, 24), ImageDecodeError, id="png-zero-width"),
        pytest.param(png_header(32, 0), ImageDecodeError, id="png-zero-height"),
        pytest.param(jpeg_header(0, 24), ImageDecodeError, id="jpeg-zero-width"),
        pytest.param(jpeg_header(32, 0), ImageDecodeError, id="jpeg-zero-height"),
        # Pillow reads a BMP width as unsigned, so -32 claims about 4.3e9 pixels.
        pytest.param(_bmp_with_width(-32), ImageTooLargeError, id="bmp-negative-width"),
    ],
)
def test_malformed_or_invalid_headers_fail_with_a_client_error(data: bytes, error: type) -> None:
    with pytest.raises(ImageDecodeError) as info:
        check_image_dimensions(data)
    assert type(info.value) is error


@pytest.mark.parametrize("size", [(0, 24), (32, 0), (-1, 24)])
def test_rejects_non_positive_dimensions_from_the_parser(
    monkeypatch: pytest.MonkeyPatch, size: tuple[int, int]
) -> None:
    # Pillow already refuses these headers; the guard checks again on its own.
    class _Parsed:
        def __init__(self) -> None:
            self.size = size

        def __enter__(self) -> _Parsed:
            return self

        def __exit__(self, *exc_info: object) -> None:
            return None

    monkeypatch.setattr(image_module.Image, "open", lambda *args, **kwargs: _Parsed())
    with pytest.raises(ImageDecodeError) as info:
        check_image_dimensions(b"stub")
    assert type(info.value) is ImageDecodeError


@pytest.mark.parametrize("fmt", ["GIF", "WEBP", "TIFF"])
def test_rejects_formats_outside_the_header_check(fmt: str) -> None:
    # A GIF is decoded at its largest frame's size, which its header doesn't
    # give, and opening a WebP allocates its canvas, so neither is parsed.
    # TF 2.15 can't decode WebP or TIFF anyway.
    with pytest.raises(ImageDecodeError) as info:
        check_image_dimensions(encode(fmt))
    assert type(info.value) is ImageDecodeError


def test_bytes_to_tensor_checks_dimensions_before_importing_tensorflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # With TensorFlow unimportable, reaching the decoder raises ImportError.
    monkeypatch.setitem(sys.modules, "tensorflow", None)

    with pytest.raises(ImageTooLargeError):
        bytes_to_tensor(png_header(10_000, 5_000))
    with pytest.raises(ImageDecodeError):
        bytes_to_tensor(b"not an image at all")
    with pytest.raises(ImportError):
        bytes_to_tensor(encode("PNG"))  # within the limits: goes on to decode
