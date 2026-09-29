"""End-to-end recognition of the private development photo with the real models.

Opt-in only (``RUN_OCR_TESTS=1``: loads PaddleOCR, ~10 s) and skipped when
the local photo is missing, so CI never depends on either.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.db.session import Database
from app.main import create_app
from app.schemas.recognition import RecognitionStatus
from app.services.card_detection.debug import DebugImageWriter
from app.services.card_detection.detector import FULL_IMAGE_METHOD
from app.services.card_detection.orientation import rotate_image
from app.services.ocr.paddle_provider import PaddleOCRProvider
from app.services.providers.json_file import JsonFileProvider
from app.services.recognition.factory import build_recognition_service
from app.services.recognition.image_io import load_image_file
from app.services.sync import SyncService

pytestmark = [
    pytest.mark.integration,
    pytest.mark.ocr,
    pytest.mark.skipif(os.environ.get("RUN_OCR_TESTS") != "1", reason="set RUN_OCR_TESTS=1 to run the real OCR model"),
]

ARMADES = "Armades, Keeper of Boundaries"


@pytest.fixture(scope="module")
def paddle() -> PaddleOCRProvider:
    provider = PaddleOCRProvider()
    provider.warmup()
    return provider


@pytest.fixture()
def paddle_settings(test_settings: Settings, fixtures_dir: Path) -> Settings:
    """Isolated settings with the fixture cards synced and the real OCR provider selected."""
    database = Database(test_settings.resolved_database_url)
    try:
        report = SyncService(database, JsonFileProvider(fixtures_dir / "ygoprodeck_sample.json"), test_settings).sync_all()
        assert report.cards_created == 7, report.summary()
    finally:
        database.dispose()
    test_settings.ocr_provider = "paddle"
    return test_settings


def test_pipeline_matches_the_local_photo(
    local_test_photo: Path, paddle_settings: Settings, paddle: PaddleOCRProvider, tmp_path: Path
) -> None:
    database = Database(paddle_settings.resolved_database_url)
    try:
        service = build_recognition_service(paddle_settings, database, ocr=paddle)
        writer = DebugImageWriter(tmp_path / "debug")
        result = service.recognize(load_image_file(local_test_photo), debug=writer)
    finally:
        database.dispose()

    assert result.status is RecognitionStatus.MATCHED, result.model_dump()
    assert result.card is not None and result.card.name == ARMADES
    assert result.printing is not None
    assert result.printing.set_code == "JOTL-EN045"
    assert result.printing.set_name == "Judgment of the Light"
    assert result.printing.rarity == "Secret Rare"
    assert result.confidence >= 0.8
    assert result.ocr.set_code is not None and result.ocr.set_code.normalized == "JOTL-EN045"
    assert result.ocr.name is not None and "armades" in result.ocr.name.normalized
    assert result.detection is not None and result.detection.detected
    assert result.detection.method != FULL_IMAGE_METHOD and result.detection.rotation_applied == 0
    assert {"04_name_roi_0", "05_set_code_roi_0", "02_normalized_card"} <= set(result.debug or {})
    assert result.timing_ms["total"] < 20_000  # sanity bound (model already loaded), not a benchmark


def test_pipeline_matches_the_photo_upside_down(
    local_test_photo: Path, paddle_settings: Settings, paddle: PaddleOCRProvider
) -> None:
    database = Database(paddle_settings.resolved_database_url)
    try:
        service = build_recognition_service(paddle_settings, database, ocr=paddle)
        result = service.recognize(rotate_image(load_image_file(local_test_photo), 180))
    finally:
        database.dispose()

    assert result.status is RecognitionStatus.MATCHED, result.model_dump()
    assert result.printing is not None and result.printing.set_code == "JOTL-EN045"
    assert result.detection is not None and result.detection.rotation_applied == 180


def test_api_recognizes_the_local_photo(
    local_test_photo: Path, paddle_settings: Settings, paddle: PaddleOCRProvider
) -> None:
    app = create_app(paddle_settings)
    with TestClient(app) as client:
        # Reuse the already loaded model instead of loading a second copy.
        app.state.service = build_recognition_service(paddle_settings, app.state.database, ocr=paddle)
        with local_test_photo.open("rb") as handle:
            response = client.post("/api/recognize", files={"image": ("172026.jpg", handle, "image/jpeg")})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "MATCHED"
    assert body["card"]["name"] == ARMADES
    assert body["printing"]["set_code"] == "JOTL-EN045"
    assert body["printing"]["set_name"] == "Judgment of the Light"
    assert body["printing"]["rarity"] == "Secret Rare"
