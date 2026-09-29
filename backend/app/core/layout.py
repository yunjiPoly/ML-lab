"""Card geometry and region-of-interest (ROI) templates.

All coordinates are expressed in the *canonical* normalized card space
(``card_width`` x ``card_height``, default 421 x 614, portrait, upright).  When
the pipeline warps the card at a higher resolution for OCR it scales the ROI
with :meth:`Roi.scaled`.

Templates exist so that cards with different layouts (classic, pendulum, ...)
can get their own regions later without touching the pipeline.  The initial
values come from measurements on one real photograph and are starting points,
not constants.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Physical Yu-Gi-Oh! card: 59 mm x 86 mm -> aspect ratio (w/h) ~ 0.686.
CARD_ASPECT_RATIO: float = 59.0 / 86.0
CARD_ASPECT_MIN: float = 0.55
CARD_ASPECT_MAX: float = 0.80

DEFAULT_CARD_WIDTH: int = 421
DEFAULT_CARD_HEIGHT: int = 614


@dataclass(frozen=True)
class Roi:
    """Axis-aligned rectangle in canonical card coordinates (x2/y2 exclusive)."""

    key: str
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def scaled(self, factor: float) -> "Roi":
        """Return the same ROI scaled by ``factor`` (used for high-res OCR warps)."""
        return Roi(
            self.key,
            int(round(self.x1 * factor)),
            int(round(self.y1 * factor)),
            int(round(self.x2 * factor)),
            int(round(self.y2 * factor)),
        )

    def clamp(self, width: int, height: int) -> "Roi":
        return Roi(
            self.key,
            max(0, min(self.x1, width)),
            max(0, min(self.y1, height)),
            max(0, min(self.x2, width)),
            max(0, min(self.y2, height)),
        )

    def crop(self, image: np.ndarray) -> np.ndarray:
        """Crop ``image`` (H x W x C) to this ROI, clamped to the image bounds."""
        h, w = image.shape[:2]
        r = self.clamp(w, h)
        return image[r.y1 : r.y2, r.x1 : r.x2]

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True)
class CardLayoutTemplate:
    """Where to look for text on a normalized card.

    ``name_rois`` and ``set_code_rois`` are ordered by preference; the OCR stage
    tries them in order and keeps the best plausible reading.
    """

    key: str
    description: str
    name_rois: tuple[Roi, ...]
    set_code_rois: tuple[Roi, ...]
    width: int = DEFAULT_CARD_WIDTH
    height: int = DEFAULT_CARD_HEIGHT

    @property
    def primary_name_roi(self) -> Roi:
        return self.name_rois[0]

    @property
    def primary_set_code_roi(self) -> Roi:
        return self.set_code_rois[0]

    def scaled(self, factor: float) -> "CardLayoutTemplate":
        return CardLayoutTemplate(
            key=self.key,
            description=self.description,
            name_rois=tuple(r.scaled(factor) for r in self.name_rois),
            set_code_rois=tuple(r.scaled(factor) for r in self.set_code_rois),
            width=int(round(self.width * factor)),
            height=int(round(self.height * factor)),
        )


STANDARD_LAYOUT = CardLayoutTemplate(
    key="standard",
    description="Modern (2002+) Monster / Spell / Trap layout. Set code right-aligned below the artwork.",
    name_rois=(
        Roi("name", 15, 10, 360, 65),
    ),
    set_code_rois=(
        # Measured on a real JOTL-EN045 photo.
        Roi("set_code", 290, 435, 405, 465),
        # Slightly wider/taller band for cards whose code sits a few px off.
        Roi("set_code_wide", 250, 428, 410, 472),
    ),
)

PENDULUM_LAYOUT = CardLayoutTemplate(
    key="pendulum",
    description="Pendulum Monster layout (experimental: set code printed at the bottom-left, "
    "below the Pendulum text box). Values are untuned placeholders.",
    name_rois=(
        Roi("name", 15, 10, 360, 65),
    ),
    set_code_rois=(
        Roi("set_code_pendulum", 15, 500, 190, 532),
        Roi("set_code", 290, 435, 405, 465),
    ),
)

LAYOUTS: dict[str, CardLayoutTemplate] = {
    STANDARD_LAYOUT.key: STANDARD_LAYOUT,
    PENDULUM_LAYOUT.key: PENDULUM_LAYOUT,
}


def get_layout(key: str = "standard") -> CardLayoutTemplate:
    """Return the ROI template registered under ``key`` (KeyError if unknown)."""
    try:
        return LAYOUTS[key]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"Unknown card layout '{key}'. Known: {sorted(LAYOUTS)}") from exc
