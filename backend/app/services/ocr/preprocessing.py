"""Pure image-preprocessing helpers for OCR crops.

Every function takes a ``numpy`` image (``H x W`` gray or ``H x W x 3`` BGR,
``uint8``) and returns a new array; nothing here touches the OCR engine, files
or global state, so each step can be unit-tested in isolation and freely
composed by :mod:`app.services.ocr.variants`.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import reduce

import cv2
import numpy as np

ImageFn = Callable[[np.ndarray], np.ndarray]


def _as_uint8(image: np.ndarray) -> np.ndarray:
    """Clip/convert any numeric array to ``uint8`` (floats in 0..1 are rescaled)."""
    if image.dtype == np.uint8:
        return image
    array = np.asarray(image, dtype=np.float64)
    if array.size and array.max() <= 1.0 and array.min() >= 0.0:
        array = array * 255.0
    return np.clip(np.rint(array), 0, 255).astype(np.uint8)


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    """Return a 3-channel ``uint8`` BGR image whatever the input channel layout.

    Gray (``H x W`` or ``H x W x 1``) is replicated, BGRA has its alpha dropped
    and non-``uint8`` dtypes are converted.  A 3-channel ``uint8`` input is
    returned unchanged (no copy).
    """
    image = _as_uint8(np.asarray(image))
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3:
        raise ValueError(f"Expected a 2-D or 3-D image, got shape {image.shape}")
    channels = image.shape[2]
    if channels == 1:
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    if channels == 3:
        return image
    if channels == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    raise ValueError(f"Unsupported channel count {channels}")


def to_gray(image: np.ndarray) -> np.ndarray:
    """Return a single-channel ``uint8`` image (``H x W``)."""
    image = _as_uint8(np.asarray(image))
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    return cv2.cvtColor(ensure_bgr(image), cv2.COLOR_BGR2GRAY)


def upscale(image: np.ndarray, factor: float, interpolation: int = cv2.INTER_CUBIC) -> np.ndarray:
    """Resize by ``factor`` (>= 1 enlarges).  ``factor == 1`` returns a copy."""
    if factor <= 0:
        raise ValueError("factor must be positive")
    if factor == 1:
        return image.copy()
    h, w = image.shape[:2]
    size = (max(1, int(round(w * factor))), max(1, int(round(h * factor))))
    return cv2.resize(image, size, interpolation=interpolation)


def normalize_height(image: np.ndarray, target_height: int = 64, *, allow_downscale: bool = False) -> np.ndarray:
    """Rescale so the image is ``target_height`` pixels tall, preserving aspect ratio.

    Enlarging uses bicubic interpolation; shrinking (only when
    ``allow_downscale`` is true) uses area interpolation.  Images already taller
    than the target are returned as a copy unless down-scaling is allowed.
    """
    h = image.shape[0]
    if h == 0 or h == target_height:
        return image.copy()
    if h > target_height and not allow_downscale:
        return image.copy()
    factor = target_height / float(h)
    interpolation = cv2.INTER_CUBIC if factor > 1 else cv2.INTER_AREA
    return upscale(image, factor, interpolation=interpolation)


def clahe(gray: np.ndarray, clip_limit: float = 2.0, tile: tuple[int, int] = (8, 8)) -> np.ndarray:
    """Contrast-limited adaptive histogram equalization on a gray image."""
    gray = to_gray(gray)
    equalizer = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile)
    return equalizer.apply(gray)


def unsharp_mask(image: np.ndarray, *, amount: float = 1.0, sigma: float = 1.0) -> np.ndarray:
    """Classic unsharp masking: ``sharp = img + amount * (img - blur(img))``."""
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
    return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)


def sharpen(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Kernel-based sharpening (Laplacian style); ``strength`` scales the effect."""
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    identity = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
    mixed = identity + strength * (kernel - identity)
    return cv2.filter2D(image, -1, mixed)


def denoise(image: np.ndarray, strength: int = 5) -> np.ndarray:
    """Edge-preserving bilateral filter (keeps glyph edges crisp, flattens paper grain)."""
    return cv2.bilateralFilter(image, d=strength, sigmaColor=40, sigmaSpace=40)


def otsu_threshold(gray: np.ndarray) -> np.ndarray:
    """Global Otsu binarization (keeps polarity: dark stays dark)."""
    _, binary = cv2.threshold(to_gray(gray), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def adaptive_threshold(gray: np.ndarray, block_size: int = 31, c: int = 10) -> np.ndarray:
    """Gaussian adaptive binarization; robust to uneven lighting across a crop."""
    block_size = max(3, block_size | 1)  # must be odd and > 1
    return cv2.adaptiveThreshold(
        to_gray(gray), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c
    )


def invert(image: np.ndarray) -> np.ndarray:
    """Photometric negative (``255 - image``)."""
    return cv2.bitwise_not(_as_uint8(image))


def gamma(image: np.ndarray, value: float) -> np.ndarray:
    """Gamma correction (``value < 1`` brightens mid-tones, ``> 1`` darkens)."""
    if value <= 0:
        raise ValueError("gamma must be positive")
    table = (np.linspace(0, 1, 256) ** value * 255).astype(np.uint8)
    return cv2.LUT(_as_uint8(image), table)


def border_color(image: np.ndarray, width: int = 2) -> tuple[int, ...]:
    """Median colour of the outer ``width`` pixel frame (what padding should look like)."""
    h, w = image.shape[:2]
    width = max(1, min(width, h, w))
    top, bottom = image[:width], image[h - width :]
    left, right = image[:, :width], image[:, w - width :]
    parts = [p.reshape(-1, image.shape[2]) if image.ndim == 3 else p.reshape(-1, 1) for p in (top, bottom, left, right)]
    stacked = np.concatenate(parts, axis=0)
    median = np.median(stacked, axis=0)
    return tuple(int(v) for v in np.atleast_1d(median))


def pad(image: np.ndarray, px: int | tuple[int, int], value: int | tuple[int, ...] | None = None) -> np.ndarray:
    """Add a constant border.

    ``px`` is either a single size for all sides or ``(vertical, horizontal)``.
    ``value`` defaults to the median border colour so glyphs at the crop edge
    are not clipped by a spurious black frame.
    """
    vertical, horizontal = (px, px) if isinstance(px, int) else px
    if value is None:
        value = border_color(image)
    if isinstance(value, tuple) and image.ndim == 2:
        value = value[0]
    if isinstance(value, tuple) and len(value) == 1:
        value = value[0]
    return cv2.copyMakeBorder(image, vertical, vertical, horizontal, horizontal, cv2.BORDER_CONSTANT, value=value)


def is_light_on_dark(image: np.ndarray) -> bool:
    """True when the crop looks like light text on a dark background.

    The larger Otsu class is taken to be the background; if that class is dark,
    the text is light.  Used for Spell/Trap/Xyz/Link name strips.
    """
    gray = to_gray(image)
    if gray.size == 0:
        return False
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_fraction = float(np.mean(gray <= threshold))
    if dark_fraction <= 0.5:
        return False
    background_level = float(np.median(gray[gray <= threshold]))
    return background_level < 128


def auto_polarity(image: np.ndarray) -> np.ndarray:
    """Invert the image when it is light-on-dark so the text is always dark-on-light."""
    return invert(image) if is_light_on_dark(image) else image.copy()


def tighten_vertical(image: np.ndarray, *, margin: int = 4, ink_fraction: float = 0.02) -> np.ndarray:
    """Trim rows above/below the text line based on the horizontal ink profile.

    Rows whose share of "ink" pixels (darker than Otsu after polarity fix) is
    below ``ink_fraction`` are considered margin.  ``margin`` rows are kept on
    each side.  Returns a copy of the input when nothing convincing is found.
    """
    gray = to_gray(auto_polarity(image))
    h = gray.shape[0]
    if h < 2 * margin + 4:
        return image.copy()
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = (gray < threshold).mean(axis=1)
    rows = np.flatnonzero(ink >= ink_fraction)
    if rows.size == 0:
        return image.copy()
    y1 = max(0, int(rows[0]) - margin)
    y2 = min(h, int(rows[-1]) + 1 + margin)
    if y2 - y1 < 8:
        return image.copy()
    return image[y1:y2].copy()


def pipeline(*steps: ImageFn) -> ImageFn:
    """Compose image functions left-to-right into a single callable."""

    def run(image: np.ndarray) -> np.ndarray:
        return reduce(lambda acc, fn: fn(acc), steps, image)

    return run
