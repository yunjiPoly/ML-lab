"""Integration test: card geometry on the private real-card photo.

Skips automatically (via the ``local_test_photo`` fixture) when the photo is
not present, so CI never depends on it.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.core.config import Settings
from app.core.layout import get_layout
from app.services.card_detection.detector import FULL_IMAGE_METHOD, CardDetector
from app.services.card_detection.orientation import TextDensityOrientationCorrector, rotate_image
from app.services.card_detection.roi import extract_roi_crops
from app.services.card_detection.service import CardGeometryService

pytestmark = pytest.mark.integration


@pytest.fixture()
def photo(local_test_photo: Path) -> np.ndarray:
    image = cv2.imread(str(local_test_photo))
    assert image is not None, f"could not decode {local_test_photo}"
    return image


def test_detector_finds_the_card_in_the_expected_region(photo: np.ndarray) -> None:
    height, width = photo.shape[:2]
    detection = CardDetector().detect(photo)
    assert detection is not None
    assert detection.method != FULL_IMAGE_METHOD
    assert detection.score > 0.6
    xs, ys = detection.corners[:, 0], detection.corners[:, 1]
    # The card occupies roughly x 700-1470 / y 200-1350 of the 2048x1536 photo.
    assert 0.30 * width <= xs.min() <= 0.42 * width
    assert 0.66 * width <= xs.max() <= 0.80 * width
    assert 0.08 * height <= ys.min() <= 0.20 * height
    assert 0.83 * height <= ys.max() <= 0.95 * height


def test_geometry_service_normalizes_the_real_photo(photo: np.ndarray, test_settings: Settings) -> None:
    service = CardGeometryService(test_settings)
    card = service.process(photo)
    assert card.image.shape == (test_settings.card_height, test_settings.card_width, 3)
    assert card.hires.shape == (
        test_settings.card_height * test_settings.ocr_scale,
        test_settings.card_width * test_settings.ocr_scale,
        3,
    )
    assert card.rotation_applied == 0  # the photo is already upright
    assert card.detection is not None and card.detection.method != FULL_IMAGE_METHOD

    top, bottom = TextDensityOrientationCorrector().band_scores(card.image)
    assert bottom > top

    crops = extract_roi_crops(card.hires, get_layout(test_settings.card_layout), card.scale)
    set_code = crops["set_code"][0].image
    assert set_code.shape[0] > 0 and set_code.shape[1] > 0
    assert float(cv2.cvtColor(set_code, cv2.COLOR_BGR2GRAY).std()) > 15  # contains printed text, not a flat area


def test_geometry_service_corrects_an_upside_down_photo(photo: np.ndarray, test_settings: Settings) -> None:
    card = CardGeometryService(test_settings).process(rotate_image(photo, 180))
    assert card.rotation_applied == 180
    assert card.image.shape == (test_settings.card_height, test_settings.card_width, 3)
