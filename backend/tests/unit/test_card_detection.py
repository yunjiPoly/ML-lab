"""Unit tests for the OpenCV card-detection package (synthetic images only)."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.core.config import Settings
from app.core.layout import get_layout
from app.services.card_detection.debug import DebugImageWriter, draw_detection, draw_rois
from app.services.card_detection.detector import (
    FULL_IMAGE_METHOD,
    CardDetector,
    order_points,
    quad_dimensions,
)
from app.services.card_detection.orientation import TextDensityOrientationCorrector, rotate_image
from app.services.card_detection.roi import extract_roi_crops
from app.services.card_detection.service import CardGeometryService
from app.services.card_detection.types import CardDetection, CardNotDetectedError
from app.services.card_detection.warp import normalize_card, quad_is_landscape, warp_card

CARD_W, CARD_H = 421, 614


# ------------------------------------------------------------------ synthetic helpers
def rotated_rectangle(center: tuple[float, float], width: float, height: float, degrees: float) -> np.ndarray:
    """Corners TL, TR, BR, BL of a rectangle rotated clockwise (image coordinates)."""
    theta = math.radians(degrees)
    cos, sin = math.cos(theta), math.sin(theta)
    base = np.array(
        [[-width / 2, -height / 2], [width / 2, -height / 2], [width / 2, height / 2], [-width / 2, height / 2]],
        dtype=np.float32,
    )
    rotation = np.array([[cos, -sin], [sin, cos]], dtype=np.float32)
    return (base @ rotation.T + np.array(center, dtype=np.float32)).astype(np.float32)


def synthetic_scene(degrees: float = 12.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A light card with a dark border on a noisy, blob-covered background."""
    rng = np.random.default_rng(seed)
    width, height = 900, 700
    scene = rng.integers(60, 180, size=(height, width, 3), dtype=np.uint8)
    scene = cv2.GaussianBlur(scene, (3, 3), 0)
    for _ in range(40):
        x, y, r = int(rng.integers(0, width)), int(rng.integers(0, height)), int(rng.integers(5, 40))
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.circle(scene, (x, y), r, color, -1)
    corners = rotated_rectangle((width / 2, height / 2), 300, 437, degrees)  # aspect ~0.686
    polygon = [np.round(corners).astype(np.int32)]
    cv2.fillPoly(scene, polygon, (235, 235, 235))
    cv2.polylines(scene, polygon, True, (25, 25, 25), 5)
    return scene, corners


def synthetic_card() -> np.ndarray:
    """Upright card: one-line name at the top, smooth artwork, dense small text at the bottom."""
    card = np.full((CARD_H, CARD_W, 3), 235, dtype=np.uint8)
    cv2.putText(card, "ARMADES, KEEPER", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2)
    gradient = np.linspace(70, 190, 340, dtype=np.uint8)[None, :].repeat(320, axis=0)
    card[100:420, 40:380] = np.stack([gradient, gradient // 2, 255 - gradient], axis=-1)
    rng = np.random.default_rng(1)
    alphabet = list("abcdefghijklmnopqrstuvwxyz   ")
    y = 470
    for _ in range(6):
        line = "".join(rng.choice(alphabet, size=48))
        cv2.putText(card, line, (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1)
        y += 18
    cv2.putText(card, "ATK/2300 DEF/1500", (220, 590), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)
    return card


def uniform_image(width: int, height: int, value: int = 128) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


# ------------------------------------------------------------------------ order_points
def test_order_points_restores_tl_tr_br_bl_from_shuffled_corners() -> None:
    truth = rotated_rectangle((100, 100), 60, 90, 12)
    for permutation in ([2, 0, 3, 1], [3, 2, 1, 0], [1, 3, 0, 2]):
        ordered = order_points(truth[permutation])
        assert ordered.dtype == np.float32
        np.testing.assert_allclose(ordered, truth, atol=1e-4)


def test_order_points_accepts_lists_and_rejects_wrong_counts() -> None:
    ordered = order_points([[10, 10], [50, 12], [48, 80], [8, 78]])
    assert ordered.shape == (4, 2)
    np.testing.assert_allclose(ordered[0], [10, 10])
    with pytest.raises(ValueError):
        order_points([[0, 0], [1, 0], [1, 1]])


def test_quad_dimensions_of_axis_aligned_rectangle() -> None:
    width, height = quad_dimensions(rotated_rectangle((50, 50), 30, 40, 0))
    assert (width, height) == pytest.approx((30.0, 40.0))


# ----------------------------------------------------------------------- detection
def test_detect_finds_rotated_card_on_noisy_background() -> None:
    scene, truth = synthetic_scene(degrees=12.0)
    detection = CardDetector().detect(scene)
    assert detection is not None
    assert detection.method != FULL_IMAGE_METHOD
    assert detection.corners.shape == (4, 2)
    assert detection.corners.dtype == np.float32
    errors = np.linalg.norm(detection.corners - truth, axis=1)
    assert errors.max() <= 6.0, f"corner errors too large: {errors}"
    assert 0.5 < detection.score <= 1.0
    assert detection.details["candidates"] >= 1


def test_normalize_card_returns_canonical_and_hires_portrait_shapes() -> None:
    scene, _ = synthetic_scene()
    detection = CardDetector().detect(scene)
    assert detection is not None
    canonical, hires = normalize_card(scene, detection, width=CARD_W, height=CARD_H, scale=2)
    assert canonical.shape == (CARD_H, CARD_W, 3)
    assert hires.shape == (CARD_H * 2, CARD_W * 2, 3)
    # The interior of the warped card is the light fill of the synthetic card.
    assert canonical[40:-40, 40:-40].mean() > 200


def test_landscape_quad_becomes_portrait_with_left_side_on_top() -> None:
    image = uniform_image(800, 600, 90)
    corners = rotated_rectangle((400, 300), 437, 300, 0)  # landscape quad
    cv2.fillPoly(image, [np.round(corners).astype(np.int32)], (230, 230, 230))
    x1, y1 = int(corners[0][0]), int(corners[0][1])
    y2 = int(corners[3][1])
    cv2.rectangle(image, (x1 + 10, y1 + 10), (x1 + 60, y2 - 10), (0, 0, 255), -1)  # red strip on the left
    assert quad_is_landscape(corners)
    detection = CardDetection(corners=corners, method="test", score=1.0)
    canonical, hires = normalize_card(image, detection, width=CARD_W, height=CARD_H, scale=2)
    assert canonical.shape == (CARD_H, CARD_W, 3)
    assert hires.shape == (CARD_H * 2, CARD_W * 2, 3)
    top = canonical[20:60, 60:360]
    assert top[..., 2].mean() > 200 and top[..., 0].mean() < 60  # red ended up at the top
    assert canonical[300:560, 60:360].mean() > 200  # the rest is the light card fill
    assert hires[40:120, 120:720][..., 2].mean() > 200


def test_warp_card_maps_ordered_corners_to_canvas_corners() -> None:
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    image[50:150, 30:130] = (0, 255, 0)
    corners = np.array([[30, 50], [129, 50], [129, 149], [30, 149]], dtype=np.float32)
    warped = warp_card(image, corners, 40, 60)
    assert warped.shape == (60, 40, 3)
    assert warped[5:-5, 5:-5, 1].min() > 200
    with pytest.raises(ValueError):
        warp_card(image, corners, 1, 60)


def test_full_image_fallback_for_card_proportioned_image_without_contours() -> None:
    blank = uniform_image(CARD_W, CARD_H)
    detection = CardDetector().detect(blank)
    assert detection is not None
    assert detection.method == FULL_IMAGE_METHOD
    assert detection.score < 0.5
    expected = np.array([[0, 0], [CARD_W - 1, 0], [CARD_W - 1, CARD_H - 1], [0, CARD_H - 1]], dtype=np.float32)
    np.testing.assert_array_equal(detection.corners, expected)


def test_no_fallback_when_disabled_or_proportions_are_wrong() -> None:
    assert CardDetector(allow_full_image_fallback=False).detect(uniform_image(CARD_W, CARD_H)) is None
    assert CardDetector().detect(uniform_image(600, 600)) is None
    with pytest.raises(ValueError):
        CardDetector().detect(np.zeros((0, 0, 3), dtype=np.uint8))


# ---------------------------------------------------------------------- orientation
def test_rotate_image_round_trips() -> None:
    rng = np.random.default_rng(3)
    image = rng.integers(0, 255, size=(37, 53, 3), dtype=np.uint8)
    rotated = rotate_image(image, 90)
    assert rotated.shape == (53, 37, 3)
    np.testing.assert_array_equal(rotate_image(rotated, 270), image)
    np.testing.assert_array_equal(rotate_image(rotate_image(image, 180), 180), image)
    np.testing.assert_array_equal(rotate_image(image, 450), rotated)
    zero = rotate_image(image, 0)
    np.testing.assert_array_equal(zero, image)
    assert zero is not image
    with pytest.raises(ValueError):
        rotate_image(image, 45)


def test_text_density_corrector_keeps_upright_and_fixes_flipped_card() -> None:
    card = synthetic_card()
    corrector = TextDensityOrientationCorrector()
    top, bottom = corrector.band_scores(card)
    assert bottom > top

    upright, degrees = corrector.correct(card)
    assert degrees == 0
    np.testing.assert_array_equal(upright, card)

    flipped = rotate_image(card, 180)
    fixed, degrees = corrector.correct(flipped)
    assert degrees == 180
    np.testing.assert_array_equal(fixed, card)


def test_text_density_corrector_validates_band_ratio() -> None:
    with pytest.raises(ValueError):
        TextDensityOrientationCorrector(band_ratio=0.9)


# ------------------------------------------------------------------------------ roi
def test_extract_roi_crops_keys_order_and_shapes() -> None:
    layout = get_layout("standard")
    scale = 2
    hires = np.zeros((CARD_H * scale, CARD_W * scale, 3), dtype=np.uint8)
    crops = extract_roi_crops(hires, layout, scale)
    assert list(crops) == ["name", "set_code"]
    assert [c.roi.key for c in crops["name"]] == [r.key for r in layout.name_rois]
    assert [c.roi.key for c in crops["set_code"]] == [r.key for r in layout.set_code_rois]
    scaled = layout.scaled(scale)
    for crop, roi in zip(crops["set_code"], scaled.set_code_rois, strict=True):
        assert crop.roi == roi
        assert crop.image.shape == (roi.height, roi.width, 3)
        assert crop.image.flags["C_CONTIGUOUS"]
    name = crops["name"][0]
    assert name.image.shape == (scaled.name_rois[0].height, scaled.name_rois[0].width, 3)


# ---------------------------------------------------------------------------- debug
def test_debug_image_writer_saves_named_files_and_remembers_paths(tmp_path: Path) -> None:
    writer = DebugImageWriter(tmp_path / "nested" / "dir", prefix="run1_")
    image = uniform_image(64, 48, 200)
    path = writer.save("01_x", image)
    assert path.exists() and path.name == "run1_01_x.jpg"
    mask_path = writer.save("mask", np.ones((10, 10), dtype=bool))
    assert mask_path.name == "run1_mask.jpg"
    assert writer.paths() == {"01_x": str(path), "mask": str(mask_path)}
    loaded = cv2.imread(str(path))
    assert loaded is not None and loaded.shape == (48, 64, 3)
    with pytest.raises(ValueError):
        writer.save("empty", np.zeros((0, 0, 3), dtype=np.uint8))


def test_draw_helpers_return_copies_of_the_same_shape() -> None:
    scene, corners = synthetic_scene()
    overlay = draw_detection(scene, corners)
    assert overlay.shape == scene.shape and overlay is not scene
    assert not np.array_equal(overlay, scene)
    card = synthetic_card()
    before = card.copy()
    rois = draw_rois(card, get_layout("standard"))
    assert rois.shape == card.shape
    np.testing.assert_array_equal(card, before)  # input untouched
    assert not np.array_equal(rois, card)


# -------------------------------------------------------------------------- service
def test_geometry_service_processes_synthetic_scene(test_settings: Settings, tmp_path: Path) -> None:
    scene, _ = synthetic_scene()
    service = CardGeometryService(test_settings)
    writer = DebugImageWriter(tmp_path / "debug")
    card = service.process(scene, debug=writer)
    assert card.image.shape == (test_settings.card_height, test_settings.card_width, 3)
    assert card.hires.shape == (
        test_settings.card_height * test_settings.ocr_scale,
        test_settings.card_width * test_settings.ocr_scale,
        3,
    )
    assert card.scale == test_settings.ocr_scale
    assert card.rotation_applied in (0, 180)
    assert card.detection is not None and card.detection.method != FULL_IMAGE_METHOD
    assert set(writer.paths()) == {"01_detected_contour", "02_normalized_card", "03_rois"}
    assert all(Path(p).exists() for p in writer.paths().values())


def test_geometry_service_raises_when_no_card(test_settings: Settings) -> None:
    service = CardGeometryService(test_settings)
    with pytest.raises(CardNotDetectedError, match="No card-shaped quadrilateral"):
        service.process(uniform_image(600, 600))
    with pytest.raises(ValueError):
        service.process(np.zeros((100, 100), dtype=np.uint8))
