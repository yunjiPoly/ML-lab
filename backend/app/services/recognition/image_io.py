"""Image decoding and upload validation for the recognition pipeline.

Pure helpers (no FastAPI dependency) shared by the API and the CLI:

* :func:`decode_image_bytes` turns JPEG/PNG/WEBP bytes into a BGR ``uint8``
  array, honouring the EXIF orientation tag that phone cameras write.
* :func:`validate_upload` enforces the upload size limit and the allowed MIME
  types, sniffing the magic bytes so a wrong ``Content-Type`` sent by a browser
  still works while non-images are rejected.
* :func:`downscale_to_max_side` bounds the processing time of huge photos.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.core.config import Settings

logger = logging.getLogger(__name__)

DEFAULT_MAX_SIDE = 2600
"""Longest image side kept by :func:`downscale_to_max_side` by default."""

_MIME_ALIASES: dict[str, str] = {
    "image/jpg": "image/jpeg",
    "image/pjpeg": "image/jpeg",
    "image/x-png": "image/png",
}


class ImageDecodeError(ValueError):
    """The bytes could not be decoded into an image."""


class UploadValidationError(Exception):
    """An upload was rejected before decoding.

    ``status_code`` is the HTTP status the API should answer with: 413 when the
    upload is too large, 415 when it is not an accepted image type.
    """

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def canonical_mime(content_type: str | None) -> str | None:
    """Lower-cased media type without parameters; common aliases folded (``image/jpg`` -> ``image/jpeg``)."""
    if not content_type:
        return None
    mime = content_type.split(";", 1)[0].strip().lower()
    return _MIME_ALIASES.get(mime, mime) or None


def sniff_image_mime(data: bytes) -> str | None:
    """MIME type implied by the magic bytes of ``data`` (JPEG, PNG, WEBP, GIF) or ``None``."""
    head = bytes(data[:16])
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    if head.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    return None


def validate_upload(
    content_type: str | None,
    size: int,
    settings: Settings,
    *,
    data: bytes | None = None,
) -> str:
    """Validate an upload's size and media type and return the effective MIME type.

    Raises :class:`UploadValidationError` with status 413 when ``size`` exceeds
    ``settings.max_upload_bytes`` and 415 when the type is not in
    ``settings.allowed_upload_mime_types`` (or the upload is empty).

    When ``data`` is given its magic bytes decide the type, so a generic or
    wrong ``Content-Type`` from a browser is ignored and non-image payloads are
    rejected regardless of what they claim to be.  Without ``data`` only the
    declared ``content_type`` is checked.
    """
    limit = settings.max_upload_bytes
    if size > limit:
        raise UploadValidationError(413, f"upload of {size} bytes exceeds the limit of {limit} bytes")
    if size <= 0:
        raise UploadValidationError(415, "empty upload; send a JPEG, PNG or WEBP image")
    allowed = settings.allowed_upload_mime_type_set
    accepted = ", ".join(sorted(allowed))
    declared = canonical_mime(content_type)
    if data is None:
        if declared not in allowed:
            raise UploadValidationError(415, f"unsupported media type {declared or 'unknown'}; accepted: {accepted}")
        return declared
    sniffed = sniff_image_mime(data)
    if sniffed is None:
        raise UploadValidationError(
            415,
            f"file content is not a recognizable image (declared type {declared or 'unknown'}); accepted: {accepted}",
        )
    if sniffed not in allowed:
        raise UploadValidationError(415, f"{sniffed} images are not accepted; accepted: {accepted}")
    if declared and declared != sniffed:
        logger.debug("Upload declared %s but contains %s; trusting the content", declared, sniffed)
    return sniffed


def decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode image bytes into a contiguous BGR ``uint8`` array (H x W x 3).

    The EXIF orientation tag is applied (phones store landscape sensor data
    plus a rotation flag), then the image is converted to RGB and flipped to
    the BGR channel order OpenCV expects.

    Raises:
        ImageDecodeError: the bytes are empty, truncated or not an image.
    """
    if not data:
        raise ImageDecodeError("no image data")
    try:
        with Image.open(io.BytesIO(data)) as pil_image:
            upright = ImageOps.exif_transpose(pil_image) or pil_image
            rgb = np.asarray(upright.convert("RGB"))
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ImageDecodeError(f"could not decode image: {exc}") from exc
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.size == 0:
        raise ImageDecodeError("decoded image is empty")
    return np.ascontiguousarray(rgb[:, :, ::-1])


def load_image_file(path: Path) -> np.ndarray:
    """Read and decode an image file (see :func:`decode_image_bytes`).

    Raises:
        FileNotFoundError: ``path`` is not an existing file.
        ImageDecodeError: the file is not a decodable image.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"image file not found: {file_path}")
    return decode_image_bytes(file_path.read_bytes())


def downscale_to_max_side(image: np.ndarray, max_side: int = DEFAULT_MAX_SIDE) -> np.ndarray:
    """Shrink ``image`` so its longest side is at most ``max_side`` (area interpolation).

    Images already within the bound (or ``max_side <= 0``) are returned unchanged.
    """
    height, width = image.shape[:2]
    longest = max(height, width)
    if max_side <= 0 or longest <= max_side:
        return image
    factor = max_side / float(longest)
    new_size = (max(1, int(round(width * factor))), max(1, int(round(height * factor))))
    logger.debug("Downscaling %dx%d image to %dx%d", width, height, new_size[0], new_size[1])
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
