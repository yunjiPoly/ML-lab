"""Card geometry pipeline: detect -> perspective-correct -> orient -> (debug images)."""

from __future__ import annotations

import logging

import numpy as np

from app.core.config import Settings
from app.core.layout import get_layout
from app.services.card_detection.debug import DebugImageWriter, draw_detection, draw_rois
from app.services.card_detection.detector import CardDetector
from app.services.card_detection.orientation import (
    OrientationCorrector,
    TextDensityOrientationCorrector,
    rotate_image,
)
from app.services.card_detection.types import CardNotDetectedError, NormalizedCard
from app.services.card_detection.warp import LANDSCAPE_ROTATION_DEGREES, normalize_card, quad_is_landscape

logger = logging.getLogger(__name__)

DEBUG_KEY_DETECTION = "01_detected_contour"
DEBUG_KEY_NORMALIZED = "02_normalized_card"
DEBUG_KEY_ROIS = "03_rois"


class CardGeometryService:
    """Turn a raw photo into an upright, perspective-corrected card.

    The detector and orientation corrector are injectable so tests (and a
    future OCR-based orientation step) can swap them without touching the
    pipeline.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        detector: CardDetector | None = None,
        orientation: OrientationCorrector | None = None,
    ) -> None:
        self._settings = settings
        self._detector = detector or CardDetector(
            min_area_ratio=settings.detection_min_area_ratio,
            allow_full_image_fallback=settings.allow_full_image_fallback,
        )
        self._orientation = orientation or TextDensityOrientationCorrector()
        self._layout = get_layout(settings.card_layout)

    def process(self, image: np.ndarray, *, debug: DebugImageWriter | None = None) -> NormalizedCard:
        """Detect, warp and orient the card in ``image`` (BGR, HxWx3).

        Raises:
            ValueError: the input is not a usable image array.
            CardNotDetectedError: no card-shaped quadrilateral was found.
        """
        if not isinstance(image, np.ndarray) or image.size == 0 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("process() expects a non-empty BGR image (H x W x 3 uint8 array)")
        h, w = image.shape[:2]
        detection = self._detector.detect(image)
        if detection is None:
            raise CardNotDetectedError(
                f"No card-shaped quadrilateral was found in the {w}x{h} image. Make sure the whole card "
                "is visible, in focus and contrasts with the background, or crop the photo to the card."
            )
        if debug is not None:
            debug.save(DEBUG_KEY_DETECTION, draw_detection(image, detection.corners))

        settings = self._settings
        canonical, hires = normalize_card(
            image,
            detection,
            width=settings.card_width,
            height=settings.card_height,
            scale=settings.ocr_scale,
        )
        base_rotation = LANDSCAPE_ROTATION_DEGREES if quad_is_landscape(detection.corners) else 0

        upright, degrees = self._orientation.correct(canonical)
        if degrees % 360:
            hires = rotate_image(hires, degrees)
        total_rotation = (base_rotation + degrees) % 360

        if debug is not None:
            debug.save(DEBUG_KEY_NORMALIZED, upright)
            debug.save(DEBUG_KEY_ROIS, draw_rois(upright, self._layout))

        logger.info(
            "Card normalized: method=%s score=%.3f rotation=%d canonical=%dx%d hires=%dx%d",
            detection.method, detection.score, total_rotation,
            upright.shape[1], upright.shape[0], hires.shape[1], hires.shape[0],
        )
        return NormalizedCard(
            image=upright,
            hires=hires,
            scale=settings.ocr_scale,
            rotation_applied=total_rotation,
            detection=detection,
        )
