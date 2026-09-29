"""Ordered preprocessing variants tried for each OCR field.

A variant is a ``(name, fn)`` pair where ``fn`` maps a BGR crop to another
image (gray or BGR).  The provider runs every variant through the recognizer
in a single batch and :func:`app.services.ocr.selection.select_best_reading`
picks the winner, so the order only matters for tie-breaking and debug output.

Field notes (measured on a real JOTL-EN045 photo, canonical 421x614 card,
OCR crops taken from the 2x warp):

* **name** - serif small-caps, ~25 px cap height at 2x.  Dark on a light strip
  for Monsters, but *white on a dark/coloured strip* for Spell/Trap/Xyz/Link
  cards, hence the polarity-normalizing variants.  Padding matters: the
  recognizer clips glyphs that touch the crop border (the level/attribute icon
  sits right at the end of the strip).
* **set_code** - tiny (~10-12 px x-height at 2x), dark on light.  Gentle
  upscaling + sharpening or CLAHE helps; hard thresholding hurts anti-aliased
  glyphs but is kept as one variant for very low-contrast prints.

Keep each list to 4-6 entries: every entry is one more forward pass.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import cv2
import numpy as np

from app.services.ocr import preprocessing as pp

Variant = tuple[str, Callable[[np.ndarray], np.ndarray]]

_PAD_NAME = 12
_PAD_CODE = 8


def _name_padded(image: np.ndarray) -> np.ndarray:
    return pp.pad(pp.ensure_bgr(image), (_PAD_NAME // 2, _PAD_NAME))


def _name_polarity_padded(image: np.ndarray) -> np.ndarray:
    """Force dark-on-light before padding (Spell/Trap/Xyz/Link strips)."""
    return pp.pad(pp.auto_polarity(pp.ensure_bgr(image)), (_PAD_NAME // 2, _PAD_NAME))


def _name_inverted_padded(image: np.ndarray) -> np.ndarray:
    """Unconditional negative for the cases the polarity heuristic misses."""
    return pp.pad(pp.invert(pp.ensure_bgr(image)), (_PAD_NAME // 2, _PAD_NAME))


def _name_gray_clahe_padded(image: np.ndarray) -> np.ndarray:
    gray = pp.clahe(pp.to_gray(pp.auto_polarity(pp.ensure_bgr(image))), clip_limit=2.0, tile=(4, 8))
    return pp.pad(gray, (_PAD_NAME // 2, _PAD_NAME))


def _name_upscale_sharp_padded(image: np.ndarray) -> np.ndarray:
    up = pp.upscale(pp.ensure_bgr(image), 2.0, interpolation=cv2.INTER_CUBIC)
    return pp.pad(pp.unsharp_mask(up, amount=0.8, sigma=1.5), (_PAD_NAME, _PAD_NAME * 2))


NAME_VARIANTS: list[Variant] = [
    ("raw", pp.ensure_bgr),
    ("pad", _name_padded),
    ("polarity_pad", _name_polarity_padded),
    ("invert_pad", _name_inverted_padded),
    ("clahe_pad", _name_gray_clahe_padded),
    ("up2_sharp_pad", _name_upscale_sharp_padded),
]
"""Ordered variants for the card-name strip."""


def _code_padded(image: np.ndarray) -> np.ndarray:
    return pp.pad(pp.ensure_bgr(image), _PAD_CODE)


def _code_upscale_sharp(image: np.ndarray, factor: float = 2.0) -> np.ndarray:
    up = pp.upscale(pp.ensure_bgr(image), factor, interpolation=cv2.INTER_CUBIC)
    return pp.pad(pp.unsharp_mask(up, amount=0.7, sigma=1.2), _PAD_CODE)


def _code_upscale_clahe(image: np.ndarray, factor: float = 3.0) -> np.ndarray:
    gray = pp.to_gray(pp.upscale(pp.ensure_bgr(image), factor, interpolation=cv2.INTER_CUBIC))
    return pp.pad(pp.clahe(gray, clip_limit=2.5, tile=(4, 8)), _PAD_CODE)


def _code_upscale_otsu(image: np.ndarray, factor: float = 3.0) -> np.ndarray:
    gray = pp.to_gray(pp.upscale(pp.ensure_bgr(image), factor, interpolation=cv2.INTER_CUBIC))
    return pp.pad(pp.otsu_threshold(pp.denoise(gray)), _PAD_CODE, value=255)


def _code_upscale_adaptive(image: np.ndarray, factor: float = 3.0) -> np.ndarray:
    gray = pp.to_gray(pp.upscale(pp.ensure_bgr(image), factor, interpolation=cv2.INTER_CUBIC))
    return pp.pad(pp.adaptive_threshold(gray, block_size=41, c=12), _PAD_CODE, value=255)


SET_CODE_VARIANTS: list[Variant] = [
    ("raw", pp.ensure_bgr),
    ("pad", _code_padded),
    ("up2_sharp", partial(_code_upscale_sharp, factor=2.0)),
    ("up3_clahe", partial(_code_upscale_clahe, factor=3.0)),
    ("up3_otsu", partial(_code_upscale_otsu, factor=3.0)),
    ("up3_adaptive", partial(_code_upscale_adaptive, factor=3.0)),
]
"""Ordered variants for the set-code strip."""


def variant_names(variants: list[Variant]) -> list[str]:
    """Names of ``variants`` in order (handy for logs and debug file names)."""
    return [name for name, _ in variants]
