"""Region-of-interest extraction from the high-resolution normalized card."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from app.core.layout import CardLayoutTemplate, Roi

logger = logging.getLogger(__name__)


@dataclass
class RoiCrop:
    """One cropped text region together with the (scaled) ROI it came from."""

    roi: Roi
    image: np.ndarray


def _crop(roi: Roi, image: np.ndarray) -> RoiCrop:
    return RoiCrop(roi=roi, image=np.ascontiguousarray(roi.crop(image)))


def extract_roi_crops(
    hires: np.ndarray, layout: CardLayoutTemplate, scale: int
) -> dict[str, list[RoiCrop]]:
    """Crop the name and set-code regions of ``layout`` (scaled by ``scale``) from ``hires``.

    Returns ``{"name": [...], "set_code": [...]}`` preserving the template order,
    so the OCR stage can try the ROIs by preference.
    """
    scaled = layout.scaled(scale)
    h, w = hires.shape[:2]
    if (w, h) != (scaled.width, scaled.height):
        logger.warning(
            "hires image is %dx%d but layout '%s' scaled by %d expects %dx%d; crops are clamped",
            w, h, layout.key, scale, scaled.width, scaled.height,
        )
    return {
        "name": [_crop(roi, hires) for roi in scaled.name_rois],
        "set_code": [_crop(roi, hires) for roi in scaled.set_code_rois],
    }
