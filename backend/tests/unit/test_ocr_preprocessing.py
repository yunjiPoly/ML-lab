"""Shape/dtype/behaviour tests for the pure OCR preprocessing helpers (no Paddle)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.services.ocr import preprocessing as pp


@pytest.fixture()
def bgr() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(40, 120, 3), dtype=np.uint8)


@pytest.fixture()
def gray() -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.integers(0, 256, size=(40, 120), dtype=np.uint8)


def _text_strip(*, light_background: bool, height: int = 110, width: int = 600) -> np.ndarray:
    """Synthetic name strip: vertical strokes (glyphs) in the middle rows, border line at the bottom."""
    bg, fg = (230, 20) if light_background else (25, 235)
    img = np.full((height, width), bg, dtype=np.uint8)
    for x in range(20, width - 20, 12):  # "glyph" strokes between rows 40 and 70
        img[40:70, x : x + 3] = fg
    img[height - 8 :, :] = 0  # dark horizontal frame line like a real strip border
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


# ------------------------------------------------------------ channel handling
def test_ensure_bgr_from_gray_and_bgra_and_float(gray: np.ndarray, bgr: np.ndarray) -> None:
    assert pp.ensure_bgr(gray).shape == (40, 120, 3)
    assert pp.ensure_bgr(gray[:, :, None]).shape == (40, 120, 3)
    bgra = np.dstack([bgr, np.full((40, 120), 255, np.uint8)])
    assert pp.ensure_bgr(bgra).shape == (40, 120, 3)
    as_float = bgr.astype(np.float32) / 255.0
    out = pp.ensure_bgr(as_float)
    assert out.dtype == np.uint8
    assert np.abs(out.astype(int) - bgr.astype(int)).max() <= 1
    assert pp.ensure_bgr(bgr) is bgr  # no copy for the common case


def test_ensure_bgr_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        pp.ensure_bgr(np.zeros((4, 4, 2), np.uint8))
    with pytest.raises(ValueError):
        pp.ensure_bgr(np.zeros((4,), np.uint8))


def test_to_gray(bgr: np.ndarray, gray: np.ndarray) -> None:
    assert pp.to_gray(bgr).shape == (40, 120)
    assert pp.to_gray(bgr).dtype == np.uint8
    assert pp.to_gray(gray) is gray


# ------------------------------------------------------------ geometry
def test_upscale_and_resize_to(bgr: np.ndarray) -> None:
    assert pp.upscale(bgr, 2).shape == (80, 240, 3)
    assert pp.upscale(bgr, 0.5).shape == (20, 60, 3)
    copy = pp.upscale(bgr, 1)
    assert copy is not bgr and np.array_equal(copy, bgr)
    with pytest.raises(ValueError):
        pp.upscale(bgr, 0)
    assert pp.resize_to(bgr, 300, 48).shape == (48, 300, 3)
    assert pp.resize_to(bgr, 10, 10).shape == (10, 10, 3)
    assert pp.resize_to(bgr, 120, 40) is not bgr


def test_stretch_changes_width_only(bgr: np.ndarray) -> None:
    assert pp.stretch(bgr, 2.0).shape == (40, 240, 3)
    assert pp.stretch(bgr, 0.5).shape == (40, 60, 3)
    with pytest.raises(ValueError):
        pp.stretch(bgr, -1)


def test_normalize_height(bgr: np.ndarray) -> None:
    assert pp.normalize_height(bgr, 80).shape == (80, 240, 3)
    assert pp.normalize_height(bgr, 20).shape == (40, 120, 3)  # no downscale by default
    assert pp.normalize_height(bgr, 20, allow_downscale=True).shape == (20, 60, 3)
    assert pp.normalize_height(bgr, 40) is not bgr


# ------------------------------------------------------------ photometric
def test_contrast_and_sharpen_keep_shape(bgr: np.ndarray, gray: np.ndarray) -> None:
    assert pp.clahe(gray).shape == gray.shape
    assert pp.clahe(bgr).shape == gray.shape  # accepts colour, returns gray
    assert pp.unsharp_mask(bgr).shape == bgr.shape and pp.unsharp_mask(bgr).dtype == np.uint8
    assert pp.sharpen(gray).shape == gray.shape
    assert pp.denoise(bgr).shape == bgr.shape


def test_thresholds_are_binary(gray: np.ndarray, bgr: np.ndarray) -> None:
    for out in (pp.otsu_threshold(gray), pp.adaptive_threshold(gray), pp.adaptive_threshold(bgr, block_size=10)):
        assert out.shape == gray.shape
        assert set(np.unique(out)).issubset({0, 255})


def test_invert_and_gamma(gray: np.ndarray) -> None:
    assert np.array_equal(pp.invert(gray), 255 - gray)
    assert np.array_equal(pp.gamma(gray, 1.0), gray)
    assert pp.gamma(gray, 0.5).mean() >= gray.mean()
    with pytest.raises(ValueError):
        pp.gamma(gray, 0)


# ------------------------------------------------------------ padding
def test_pad_uses_border_colour_by_default() -> None:
    img = np.full((20, 50, 3), (10, 200, 30), dtype=np.uint8)
    out = pp.pad(img, 5)
    assert out.shape == (30, 60, 3)
    assert tuple(out[0, 0]) == (10, 200, 30)
    out2 = pp.pad(img, (2, 7), value=(0, 0, 0))
    assert out2.shape == (24, 64, 3) and tuple(out2[0, 0]) == (0, 0, 0)
    gray = np.full((20, 50), 77, dtype=np.uint8)
    assert pp.pad(gray, 3).shape == (26, 56) and pp.pad(gray, 3)[0, 0] == 77
    assert pp.border_color(img) == (10, 200, 30)


# ------------------------------------------------------------ polarity
def test_polarity_detection_and_auto_invert() -> None:
    light = _text_strip(light_background=True)
    dark = _text_strip(light_background=False)
    assert pp.is_light_on_dark(light) is False
    assert pp.is_light_on_dark(dark) is True
    assert np.array_equal(pp.auto_polarity(light), light)
    assert np.array_equal(pp.auto_polarity(dark), pp.invert(dark))
    assert pp.is_light_on_dark(np.zeros((0, 0), np.uint8)) is False


# ------------------------------------------------------------ text band
@pytest.mark.parametrize("light_background", [True, False])
def test_text_row_band_finds_glyph_rows_not_border(light_background: bool) -> None:
    strip = _text_strip(light_background=light_background)
    band = pp.text_row_band(strip, margin=4)
    assert band is not None
    y1, y2 = band
    assert 30 <= y1 <= 40 and 70 <= y2 <= 80, band  # glyphs at 40..70, +-margin
    tight = pp.tighten_vertical(strip)
    assert tight.shape[0] == y2 - y1 and tight.shape[1] == strip.shape[1]


def test_tighten_vertical_is_safe_on_degenerate_input() -> None:
    flat = np.full((60, 200, 3), 128, dtype=np.uint8)
    assert pp.text_row_band(flat) is None
    assert pp.tighten_vertical(flat).shape == flat.shape
    tiny = np.zeros((5, 20, 3), dtype=np.uint8)
    assert pp.tighten_vertical(tiny).shape == tiny.shape


# ------------------------------------------------------------ composition
def test_pipeline_composes_left_to_right(bgr: np.ndarray) -> None:
    fn = pp.pipeline(pp.to_gray, pp.invert, lambda g: pp.upscale(g, 2))
    out = fn(bgr)
    assert out.shape == (80, 240)
    assert np.array_equal(out, pp.upscale(pp.invert(pp.to_gray(bgr)), 2))
