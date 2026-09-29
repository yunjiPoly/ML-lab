"""Ordered preprocessing variants tried for each OCR field.

A variant is a ``(name, fn)`` pair where ``fn`` maps a BGR crop to another
image (gray or BGR).  The provider runs the variants through the recognizer
and :func:`app.services.ocr.selection.select_best_reading` picks the winner,
so the order only matters for tie-breaking and debug output.

Why every variant ends on a shared canvas
-----------------------------------------
PaddleOCR's recognizer (PP-OCRv5/v6 via the transformers image processor)
normalizes *the whole batch* to one size: height 48 and the aspect ratio of
the widest image in the batch.  Variants of different shapes therefore
stretch or squeeze each other, and a reading would depend on which siblings
happened to be in the batch.  Each field defines a canvas derived from the
*original* crop, and every variant resizes to it as its last step, so the
batch is deterministic and identical to single-image inference.

Field notes (measured on a real JOTL-EN045 photo; see the experiment log in
the OCR provider report):

* **name** - serif small-caps, long (up to ~30 characters) in a strip with a
  ~6:1 aspect ratio.  At the recognizer's 48 px height that leaves too few
  CTC time-steps per glyph and letters get dropped ("ARMADESKEEPEOBOUNDR" at
  0.71).  Stretching the strip horizontally 2x - or cropping vertically to the
  text band - restores the full "ARMADES, KEEPER OF BOUNDARIES" at 0.98-0.997.
  Dark-on-light for Monsters but white-on-dark for Spell/Trap/Xyz/Link cards,
  hence the inverted variant.
* **set_code** - tiny (~10-12 px x-height at 2x), dark on light (white on
  black for Xyz).  Mild upscaling + sharpening or CLAHE helps; hard
  thresholding hurts anti-aliased glyphs but is kept for low-contrast prints.
  Stretching *hurts* here (a 1.5x stretched code reads "ENO45"), so its canvas
  keeps the natural aspect ratio.

Keep each list to 4-6 entries: every entry is one more forward pass.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from app.services.ocr import preprocessing as pp

Variant = tuple[str, Callable[[np.ndarray], np.ndarray]]

REC_INPUT_HEIGHT: int = 48
"""Input height of the PP-OCR recognizers; the canvas height for every variant."""

MAX_CANVAS_WIDTH: int = 1600
"""Safety cap well under PaddleOCR's 3200 px recognizer limit."""

NAME_STRETCH: float = 2.0
"""Horizontal stretch applied to the whole name strip (see module docstring)."""

SET_CODE_STRETCH: float = 1.0
"""No stretch for set codes: the digits are short and distort easily."""


def canvas_size(image: np.ndarray, stretch: float, height: int = REC_INPUT_HEIGHT) -> tuple[int, int]:
    """``(width, height)`` of the shared canvas for a crop of this shape."""
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return (1, height)
    width = int(round(height * (w / float(h)) * stretch))
    return (max(1, min(width, MAX_CANVAS_WIDTH)), height)


def _on_canvas(processed: np.ndarray, original: np.ndarray, stretch: float) -> np.ndarray:
    width, height = canvas_size(original, stretch)
    return pp.resize_to(pp.ensure_bgr(processed), width, height)


# ------------------------------------------------------------------ name
def _name_stretch(image: np.ndarray) -> np.ndarray:
    """Whole strip, horizontally stretched (the new baseline)."""
    return _on_canvas(pp.ensure_bgr(image), image, NAME_STRETCH)


def _name_stretch_sharp(image: np.ndarray) -> np.ndarray:
    """Whole strip, unsharp-masked at native resolution (slightly blurry photos)."""
    return _on_canvas(pp.unsharp_mask(pp.ensure_bgr(image), amount=0.8, sigma=1.5), image, NAME_STRETCH)


def _name_tight(image: np.ndarray) -> np.ndarray:
    """Vertically tightened to the text band; the glyphs get more of the 48 px."""
    return _on_canvas(pp.tighten_vertical(pp.ensure_bgr(image)), image, NAME_STRETCH)


def _name_tight_invert(image: np.ndarray) -> np.ndarray:
    """Tight + photometric negative (Spell/Trap/Xyz/Link white-on-dark strips)."""
    return _on_canvas(pp.invert(pp.tighten_vertical(pp.ensure_bgr(image))), image, NAME_STRETCH)


def _name_tight_clahe(image: np.ndarray) -> np.ndarray:
    """Tight + CLAHE on gray (foil / low-contrast name strips)."""
    tight = pp.tighten_vertical(pp.ensure_bgr(image))
    return _on_canvas(pp.clahe(pp.to_gray(tight), clip_limit=2.0, tile=(2, 8)), image, NAME_STRETCH)


NAME_VARIANTS: list[Variant] = [
    ("stretch", _name_stretch),
    ("tight", _name_tight),
    ("tight_invert", _name_tight_invert),
    ("tight_clahe", _name_tight_clahe),
    ("stretch_sharp", _name_stretch_sharp),
]
"""Ordered variants for the card-name strip (all on the same canvas)."""


# -------------------------------------------------------------- set code
def _code_raw(image: np.ndarray) -> np.ndarray:
    return _on_canvas(pp.ensure_bgr(image), image, SET_CODE_STRETCH)


def _code_up2_sharp(image: np.ndarray) -> np.ndarray:
    up = pp.upscale(pp.ensure_bgr(image), 2.0)
    return _on_canvas(pp.unsharp_mask(up, amount=0.7, sigma=1.2), image, SET_CODE_STRETCH)


def _code_up3_clahe(image: np.ndarray) -> np.ndarray:
    gray = pp.to_gray(pp.upscale(pp.ensure_bgr(image), 3.0))
    return _on_canvas(pp.clahe(gray, clip_limit=2.5, tile=(2, 8)), image, SET_CODE_STRETCH)


def _code_up3_otsu(image: np.ndarray) -> np.ndarray:
    gray = pp.to_gray(pp.upscale(pp.ensure_bgr(image), 3.0))
    return _on_canvas(pp.otsu_threshold(pp.denoise(gray)), image, SET_CODE_STRETCH)


def _code_up3_adaptive(image: np.ndarray) -> np.ndarray:
    gray = pp.to_gray(pp.upscale(pp.ensure_bgr(image), 3.0))
    return _on_canvas(pp.adaptive_threshold(gray, block_size=41, c=12), image, SET_CODE_STRETCH)


def _code_invert(image: np.ndarray) -> np.ndarray:
    """Negative for white-on-black codes (Xyz frames)."""
    return _on_canvas(pp.invert(pp.ensure_bgr(image)), image, SET_CODE_STRETCH)


SET_CODE_VARIANTS: list[Variant] = [
    ("raw", _code_raw),
    ("up2_sharp", _code_up2_sharp),
    ("up3_clahe", _code_up3_clahe),
    ("up3_otsu", _code_up3_otsu),
    ("up3_adaptive", _code_up3_adaptive),
    ("invert", _code_invert),
]
"""Ordered variants for the set-code strip (all on the same canvas)."""


def variant_names(variants: list[Variant]) -> list[str]:
    """Names of ``variants`` in order (handy for logs and debug file names)."""
    return [name for name, _ in variants]
