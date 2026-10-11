"""Small image byte builders for the upload dimension-guard tests.

A header-only image declares any size in a few dozen bytes, so tests can check
huge dimensions without allocating them. Nothing here decodes pixels.
"""

from __future__ import annotations

import io
import struct
import zlib

from PIL import Image


def encode(fmt: str, size: tuple[int, int] = (32, 24), mode: str = "RGB") -> bytes:
    """A small deterministic gradient image, saved as ``fmt``."""
    width, height = size
    pixels = bytes(
        (x * 7 + y * 13 + channel * 50) % 256
        for y in range(height)
        for x in range(width)
        for channel in range(3)
    )
    image = Image.frombytes("RGB", size, pixels).convert(mode)
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def png_header(width: int, height: int) -> bytes:
    """A structurally valid PNG that declares ``width`` x ``height`` and holds no pixels."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(kind + data)
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", b"") + chunk(b"IEND", b"")


def jpeg_header(width: int, height: int) -> bytes:
    """A baseline JPEG header (SOI, SOF0, SOS) for ``width`` x ``height``, with no scan data."""
    components = b"\x01\x22\x00\x02\x11\x01\x03\x11\x01"  # Y, Cb, Cr
    sof0 = b"\xff\xc0" + struct.pack(">HBHHB", 17, 8, height, width, 3) + components
    sos = b"\xff\xda" + struct.pack(">HB", 12, 3) + b"\x01\x00\x02\x11\x03\x11\x00\x3f\x00"
    return b"\xff\xd8" + sof0 + sos
