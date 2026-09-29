"""Upright-orientation correction for a perspective-normalized portrait card.

An upright Yu-Gi-Oh! card has its single-line name and the artwork in the top
part and a dense block of small effect / lore text in the bottom ~30 %.
:class:`TextDensityOrientationCorrector` measures "text-likeness" in the top and
bottom bands and rotates the card by 180 degrees when the bottom band looks
like the top one.  It is deliberately isolated behind the
:class:`OrientationCorrector` interface so an OCR-based corrector can replace
it later without touching the geometry service.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_ROTATIONS: dict[int, int] = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def rotate_image(image: np.ndarray, degrees: int) -> np.ndarray:
    """Rotate ``image`` clockwise by a multiple of 90 degrees (lossless)."""
    normalized = int(degrees) % 360
    if normalized == 0:
        return image.copy()
    try:
        code = _ROTATIONS[normalized]
    except KeyError as exc:
        raise ValueError(f"rotate_image supports multiples of 90 degrees, got {degrees}") from exc
    return cv2.rotate(image, code)


class OrientationCorrector(ABC):
    """Decide how far a portrait card must be rotated to be upright."""

    @abstractmethod
    def correct(self, card: np.ndarray) -> tuple[np.ndarray, int]:
        """Return ``(upright_card, degrees_applied)``; degrees is 0, 90, 180 or 270."""


class TextDensityOrientationCorrector(OrientationCorrector):
    """Choose between 0 and 180 degrees from the distribution of small text blobs.

    Each band is binarized with an adaptive threshold (ink = foreground) and
    split into connected components.  Components whose size is compatible with
    printed characters (relative to the card size) and that share a row with
    several similarly sized components count as text.  The count is weighted by
    the fraction of blank rows in the band, because real text lines are
    separated by empty rows while artwork noise is spread everywhere.

    Args:
        band_ratio: height of the top/bottom bands as a fraction of the card height.
        block_size: adaptive threshold neighbourhood (odd).
        threshold_c: adaptive threshold constant.
        min_row_neighbours: aligned, similar-height blobs a blob needs to count as text.
        decision_margin: the top must beat the bottom by this relative margin to flip.
    """

    def __init__(
        self,
        *,
        band_ratio: float = 0.30,
        block_size: int = 25,
        threshold_c: float = 10.0,
        min_row_neighbours: int = 6,
        decision_margin: float = 0.0,
        max_components: int = 4000,
    ) -> None:
        if not 0.05 <= band_ratio <= 0.5:
            raise ValueError("band_ratio must be within [0.05, 0.5]")
        self._band_ratio = band_ratio
        self._block_size = block_size | 1
        self._threshold_c = threshold_c
        self._min_row_neighbours = min_row_neighbours
        self._decision_margin = decision_margin
        self._max_components = max_components

    def correct(self, card: np.ndarray) -> tuple[np.ndarray, int]:
        top, bottom = self.band_scores(card)
        logger.debug("Orientation text scores: top=%.1f bottom=%.1f", top, bottom)
        if top > bottom * (1.0 + self._decision_margin):
            logger.info("Card looks upside down (top=%.1f > bottom=%.1f): rotating 180", top, bottom)
            return rotate_image(card, 180), 180
        return card, 0

    def band_scores(self, card: np.ndarray) -> tuple[float, float]:
        """Return the text scores of the top and bottom bands (higher = more text)."""
        gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY) if card.ndim == 3 else card
        height, width = gray.shape[:2]
        band = max(1, int(round(height * self._band_ratio)))
        top_score = self.text_score(gray[:band], height, width)
        bottom_score = self.text_score(gray[height - band :], height, width)
        return top_score, bottom_score

    def text_score(self, band: np.ndarray, card_height: int, card_width: int) -> float:
        """Text-likeness of a grayscale band, given the full card dimensions."""
        if band.size == 0 or min(band.shape[:2]) < self._block_size:
            return 0.0
        binary = cv2.adaptiveThreshold(
            band, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, self._block_size, self._threshold_c
        )
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if count <= 1:
            return 0.0
        stats = stats[1:]
        centroids = centroids[1:]
        widths = stats[:, cv2.CC_STAT_WIDTH].astype(np.float32)
        heights = stats[:, cv2.CC_STAT_HEIGHT].astype(np.float32)
        areas = stats[:, cv2.CC_STAT_AREA].astype(np.float32)
        max_height = max(4.0, card_height * 0.045)
        max_width = max(4.0, card_width * 0.12)
        fill = areas / np.maximum(widths * heights, 1.0)
        keep = (
            (heights >= 3.0) & (heights <= max_height) & (widths <= max_width) & (areas >= 3.0) & (fill >= 0.12)
        )
        indices = np.flatnonzero(keep)
        if indices.size == 0:
            return 0.0
        if indices.size > self._max_components:
            indices = np.random.default_rng(0).choice(indices, self._max_components, replace=False)
        heights = heights[indices]
        centers_y = centroids[indices, 1].astype(np.float32)
        dy = np.abs(centers_y[:, None] - centers_y[None, :])
        h_max = np.maximum(heights[:, None], heights[None, :])
        h_min = np.minimum(heights[:, None], heights[None, :])
        aligned = (dy <= 0.6 * h_max) & (h_min >= 0.5 * h_max)
        neighbours = aligned.sum(axis=1) - 1
        text_blobs = int(np.count_nonzero(neighbours >= self._min_row_neighbours))
        text_mask = np.isin(labels, indices + 1)
        row_ink = text_mask.mean(axis=1)
        blank_fraction = float(np.mean(row_ink < 0.005))
        return float(text_blobs * (0.2 + blank_fraction))
