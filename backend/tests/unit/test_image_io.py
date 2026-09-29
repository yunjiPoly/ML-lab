"""Unit tests for ``app.services.recognition.image_io`` (decoding, EXIF, validation)."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.core.config import Settings
from app.services.recognition.image_io import (
    ImageDecodeError,
    UploadValidationError,
    canonical_mime,
    decode_image_bytes,
    downscale_to_max_side,
    load_image_file,
    sniff_image_mime,
    validate_upload,
)

EXIF_ORIENTATION_TAG = 0x0112
ORIENTATION_ROTATE_90_CW = 6


def two_tone_rgb(width: int = 40, height: int = 20) -> Image.Image:
    """Left half pure red, right half pure blue."""
    array = np.zeros((height, width, 3), dtype=np.uint8)
    array[:, : width // 2] = (255, 0, 0)
    array[:, width // 2 :] = (0, 0, 255)
    return Image.fromarray(array, "RGB")


def encode(image: Image.Image, fmt: str, **kwargs) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **kwargs)
    return buffer.getvalue()


@pytest.fixture()
def settings() -> Settings:
    return Settings(_env_file=None, max_upload_bytes=1000, allowed_upload_mime_types="image/jpeg,image/png,image/webp")  # type: ignore[call-arg]


# ------------------------------------------------------------------ decoding
def test_decode_png_returns_contiguous_bgr_uint8() -> None:
    bgr = decode_image_bytes(encode(two_tone_rgb(), "PNG"))
    assert bgr.shape == (20, 40, 3)
    assert bgr.dtype == np.uint8
    assert bgr.flags["C_CONTIGUOUS"]
    assert tuple(bgr[10, 5]) == (0, 0, 255)  # red pixel, BGR order
    assert tuple(bgr[10, 35]) == (255, 0, 0)  # blue pixel, BGR order


def test_exif_rotated_jpeg_decodes_upright() -> None:
    exif = Image.Exif()
    exif[EXIF_ORIENTATION_TAG] = ORIENTATION_ROTATE_90_CW
    data = encode(two_tone_rgb(), "JPEG", quality=95, subsampling=0, exif=exif.tobytes())
    with Image.open(io.BytesIO(data)) as stored:
        assert stored.size == (40, 20)  # the file itself is landscape; only the tag says "rotate"

    bgr = decode_image_bytes(data)
    assert bgr.shape == (40, 20, 3)  # portrait once the orientation tag is applied
    top = bgr[:15].reshape(-1, 3).mean(axis=0)
    bottom = bgr[25:].reshape(-1, 3).mean(axis=0)
    assert top[2] > 200 and top[0] < 60  # red (left half of the stored image) ends up on top
    assert bottom[0] > 200 and bottom[2] < 60  # blue ends up at the bottom


def test_bad_bytes_raise_image_decode_error() -> None:
    with pytest.raises(ImageDecodeError):
        decode_image_bytes(b"")
    with pytest.raises(ImageDecodeError):
        decode_image_bytes(b"definitely not an image")
    with pytest.raises(ImageDecodeError):
        decode_image_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)  # PNG magic, truncated body


def test_load_image_file(tmp_path: Path) -> None:
    path = tmp_path / "card.png"
    path.write_bytes(encode(two_tone_rgb(), "PNG"))
    assert load_image_file(path).shape == (20, 40, 3)
    with pytest.raises(FileNotFoundError):
        load_image_file(tmp_path / "missing.png")
    (tmp_path / "broken.jpg").write_bytes(b"\xff\xd8\xff nope")
    with pytest.raises(ImageDecodeError):
        load_image_file(tmp_path / "broken.jpg")


def test_downscale_to_max_side() -> None:
    big = np.zeros((3000, 1500, 3), dtype=np.uint8)
    small = downscale_to_max_side(big, 600)
    assert small.shape == (600, 300, 3)
    fits = np.zeros((100, 50, 3), dtype=np.uint8)
    assert downscale_to_max_side(fits, 600) is fits
    assert downscale_to_max_side(big, 0) is big


# ---------------------------------------------------------------- validation
def test_sniff_and_canonical_mime() -> None:
    assert sniff_image_mime(encode(two_tone_rgb(), "PNG")) == "image/png"
    assert sniff_image_mime(encode(two_tone_rgb(), "JPEG")) == "image/jpeg"
    assert sniff_image_mime(encode(two_tone_rgb(), "WEBP")) == "image/webp"
    assert sniff_image_mime(b"GIF89a" + b"\x00" * 8) == "image/gif"
    assert sniff_image_mime(b"hello world") is None
    assert sniff_image_mime(b"") is None
    assert canonical_mime("image/JPG; charset=binary") == "image/jpeg"
    assert canonical_mime("image/png") == "image/png"
    assert canonical_mime(None) is None
    assert canonical_mime("") is None


def test_validate_upload_size_limits(settings: Settings) -> None:
    with pytest.raises(UploadValidationError) as too_big:
        validate_upload("image/png", 1001, settings)
    assert too_big.value.status_code == 413
    assert "1000" in too_big.value.detail
    with pytest.raises(UploadValidationError) as empty:
        validate_upload("image/png", 0, settings)
    assert empty.value.status_code == 415
    assert validate_upload("image/png", 1000, settings) == "image/png"


def test_validate_upload_declared_type_only(settings: Settings) -> None:
    assert validate_upload("image/jpg", 10, settings) == "image/jpeg"
    assert validate_upload("image/webp; boundary=x", 10, settings) == "image/webp"
    for bad in ("text/plain", "application/octet-stream", None):
        with pytest.raises(UploadValidationError) as exc:
            validate_upload(bad, 10, settings)
        assert exc.value.status_code == 415


def test_validate_upload_magic_bytes_override_declared_type(settings: Settings) -> None:
    png = encode(two_tone_rgb(), "PNG")
    assert validate_upload("application/octet-stream", len(png), settings, data=png) == "image/png"
    assert validate_upload("text/plain", len(png), settings, data=png) == "image/png"
    assert validate_upload(None, len(png), settings, data=png) == "image/png"

    with pytest.raises(UploadValidationError) as not_image:
        validate_upload("image/png", 5, settings, data=b"hello")
    assert not_image.value.status_code == 415
    assert "not a recognizable image" in not_image.value.detail

    gif = b"GIF89a" + b"\x00" * 10
    with pytest.raises(UploadValidationError) as not_allowed:
        validate_upload("image/gif", len(gif), settings, data=gif)
    assert not_allowed.value.status_code == 415
    assert "image/gif" in not_allowed.value.detail

    with pytest.raises(UploadValidationError) as too_big:
        validate_upload("image/png", 5000, settings, data=png)
    assert too_big.value.status_code == 413
