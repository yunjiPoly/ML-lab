"""Perspective correction of a detected card quadrilateral.

Both the canonical (``card_width`` x ``card_height``) image and the
higher-resolution OCR image are warped straight from the source photo, so the
hires version keeps the detail of the original pixels instead of being an
upscaled copy of the canonical one.  Landscape quadrilaterals are warped into
a landscape canvas and rotated 90 degrees clockwise so the output is always
portrait; the 0/180 ambiguity is left to the orientation corrector.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from app.services.card_detection.detector import quad_dimensions
from app.services.card_detection.orientation import rotate_image
from app.services.card_detection.types import CardDetection

logger = logging.getLogger(__name__)

LANDSCAPE_ROTATION_DEGREES: int = 90


def quad_is_landscape(corners: np.ndarray) -> bool:
    """True when the ordered quad is wider than it is tall in the source image."""
    width, height = quad_dimensions(corners)
    return width > height


def warp_card(
    image: np.ndarray,
    corners: np.ndarray,
    width: int,
    height: int,
    *,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Warp the ordered TL, TR, BR, BL quadrilateral onto a ``width`` x ``height`` canvas."""
    if width < 2 or height < 2:
        raise ValueError(f"Target size must be at least 2x2, got {width}x{height}")
    source = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(
        image, matrix, (width, height), flags=interpolation, borderMode=cv2.BORDER_REPLICATE
    )


def normalize_card(
    image: np.ndarray,
    detection: CardDetection,
    *,
    width: int,
    height: int,
    scale: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(canonical, hires)`` portrait warps of the detected card.

    ``canonical`` is ``height`` x ``width`` pixels; ``hires`` is
    ``height * scale`` x ``width * scale`` pixels, warped directly from the
    source image (bicubic) for OCR.  A landscape quad is rotated 90 degrees
    clockwise so both outputs are always portrait.
    """
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")
    corners = np.asarray(detection.corners, dtype=np.float32).reshape(4, 2)
    landscape = quad_is_landscape(corners)
    canvas_w, canvas_h = (height, width) if landscape else (width, height)
    canonical = warp_card(image, corners, canvas_w, canvas_h)
    hires = warp_card(image, corners, canvas_w * scale, canvas_h * scale, interpolation=cv2.INTER_CUBIC)
    if landscape:
        logger.debug("Detected quad is landscape: rotating the warp %d degrees", LANDSCAPE_ROTATION_DEGREES)
        canonical = rotate_image(canonical, LANDSCAPE_ROTATION_DEGREES)
        hires = rotate_image(hires, LANDSCAPE_ROTATION_DEGREES)
    return canonical, hires
