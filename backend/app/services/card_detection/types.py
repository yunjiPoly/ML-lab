"""Data types shared by the detection / warp / orientation components."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class CardNotDetectedError(Exception):
    """Raised when no plausible card quadrilateral is found."""


@dataclass
class CardDetection:
    """A detected card quadrilateral in *source image* pixel coordinates."""

    corners: np.ndarray  # shape (4, 2), float32, ordered TL, TR, BR, BL
    method: str  # detection strategy identifier
    score: float = 0.0  # geometric plausibility in [0, 1]
    details: dict = field(default_factory=dict)

    def corners_as_list(self) -> list[list[float]]:
        return [[float(x), float(y)] for x, y in self.corners.tolist()]


@dataclass
class NormalizedCard:
    """The card after perspective correction and orientation."""

    image: np.ndarray  # canonical size (card_height x card_width x 3), BGR, upright
    hires: np.ndarray  # same content at canonical size * scale for OCR crops
    scale: int  # hires / canonical scale factor
    rotation_applied: int = 0  # degrees rotated to make the card upright (0, 90, 180, 270)
    detection: CardDetection | None = None
