"""Real PaddleOCR run on the prototype crops of the local development photo.

Opt-in only: ``RUN_OCR_TESTS=1`` (loads the PP-OCRv6 model, several seconds).
Skips when the private debug crops are not present.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.core.text import normalize_card_name, normalize_set_code
from app.services.ocr.paddle_provider import PaddleOCRProvider

pytestmark = [
    pytest.mark.integration,
    pytest.mark.ocr,
    pytest.mark.skipif(os.environ.get("RUN_OCR_TESTS") != "1", reason="set RUN_OCR_TESTS=1 to run the real OCR model"),
]

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEBUG_DIR = PROJECT_ROOT / "data" / "debug"
NAME_CROP = DEBUG_DIR / "photo_name.jpg"
SET_CODE_CROP = DEBUG_DIR / "photo_set_code.jpg"


def _load(path: Path) -> np.ndarray:
    if not path.exists():
        pytest.skip(f"local crop not present: {path}")
    image = cv2.imread(str(path))
    assert image is not None, f"could not decode {path}"
    return image


@pytest.fixture(scope="module")
def provider() -> PaddleOCRProvider:
    prov = PaddleOCRProvider()
    prov.warmup()
    assert prov.is_loaded
    return prov


def test_set_code_crop_reads_jotl_en045(provider: PaddleOCRProvider) -> None:
    result = provider.recognize_set_code(_load(SET_CODE_CROP))
    assert normalize_set_code(result.text) == "JOTL-EN045", result
    assert result.confidence > 0.9
    assert result.variant is not None


def test_name_crop_reads_armades(provider: PaddleOCRProvider) -> None:
    result = provider.recognize_name(_load(NAME_CROP))
    normalized = normalize_card_name(result.text)
    assert "armades" in normalized, result
    assert "boundaries" in normalized, result  # the untreated crop truncates to "...BOUNDR"
    assert result.confidence > 0.9


def test_recognize_text_raw_crop(provider: PaddleOCRProvider) -> None:
    result = provider.recognize_text(_load(SET_CODE_CROP))
    assert normalize_set_code(result.text) == "JOTL-EN045"


def test_empty_and_tiny_crops_do_not_crash(provider: PaddleOCRProvider) -> None:
    assert provider.recognize_name(np.zeros((0, 0, 3), np.uint8)).is_empty
    tiny = np.full((3, 5, 3), 255, np.uint8)
    provider.recognize_set_code(tiny)  # any result is fine, must not raise
