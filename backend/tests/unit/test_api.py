"""FastAPI application tests (TestClient, temporary SQLite populated from the fixture, fake OCR)."""

from __future__ import annotations

import io
import math
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import Settings
from app.db.session import Database
from app.main import create_app
from app.schemas.recognition import RecognitionResult, RecognitionStatus
from app.services.providers.json_file import JsonFileProvider
from app.services.sync import SyncService

ARMADES_ID = 88033975
UPLOAD_LIMIT = 2 * 1024 * 1024


# ---------------------------------------------------------------- helpers
def synthetic_scene(degrees: float = 8.0, seed: int = 0) -> np.ndarray:
    """A light card with a dark border, slightly rotated, on a noisy grey background."""
    rng = np.random.default_rng(seed)
    width, height = 900, 700
    scene = rng.integers(70, 170, size=(height, width, 3), dtype=np.uint8)
    scene = cv2.GaussianBlur(scene, (3, 3), 0)
    theta = math.radians(degrees)
    cos, sin = math.cos(theta), math.sin(theta)
    base = np.array([[-150, -218], [150, -218], [150, 218], [-150, 218]], dtype=np.float32)  # aspect ~0.686
    rotation = np.array([[cos, -sin], [sin, cos]], dtype=np.float32)
    corners = base @ rotation.T + np.array([width / 2, height / 2], dtype=np.float32)
    polygon = [np.round(corners).astype(np.int32)]
    cv2.fillPoly(scene, polygon, (235, 235, 235))
    cv2.polylines(scene, polygon, True, (25, 25, 25), 5)
    return scene


def png_bytes(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def jpeg_bytes(size: tuple[int, int] = (64, 64)) -> bytes:
    """A flat square JPEG: a valid image, but square so the full-image card fallback does not apply."""
    buffer = io.BytesIO()
    Image.new("RGB", size, (200, 30, 30)).save(buffer, format="JPEG")
    return buffer.getvalue()


def upload(client: TestClient, data: bytes, content_type: str = "image/png", *, field: str = "image", query: str = ""):
    return client.post(f"/api/recognize{query}", files={field: ("capture.jpg", data, content_type)})


# --------------------------------------------------------------- fixtures
@pytest.fixture()
def api_settings(test_settings: Settings, fixtures_dir: Path) -> Settings:
    """Isolated settings whose SQLite database holds the 7 fixture cards."""
    test_settings.max_upload_bytes = UPLOAD_LIMIT
    database = Database(test_settings.resolved_database_url)
    try:
        provider = JsonFileProvider(fixtures_dir / "ygoprodeck_sample.json")
        report = SyncService(database, provider, test_settings).sync_all(download_images=False)
        assert report.cards_created == 7, report.summary()
    finally:
        database.dispose()
    return test_settings


@pytest.fixture()
def client(api_settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(api_settings)) as test_client:
        yield test_client


# ------------------------------------------------------------------ health
def test_health_reports_components_and_counts(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"]
    assert body["ocr_provider"] == "fake"
    assert body["visual_recognizer"] == "noop"
    assert body["database"]["cards"] == 7
    assert body["database"]["printings"] > 7


def test_cors_preflight_allows_the_dev_frontend(client: TestClient) -> None:
    response = client.options(
        "/api/health",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"},
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


# ------------------------------------------------------------------- cards
def test_get_card_returns_details_with_printings(client: TestClient) -> None:
    response = client.get(f"/api/cards/{ARMADES_ID}")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["id"] == ARMADES_ID
    assert body["name"] == "Armades, Keeper of Boundaries"
    assert body["type"] == "Synchro Monster"
    assert body["def"] == 1500
    codes = {p["set_code"] for p in body["printings"]}
    assert "JOTL-EN045" in codes
    jotl = next(p for p in body["printings"] if p["set_code"] == "JOTL-EN045")
    assert jotl["set_name"] == "Judgment of the Light" and jotl["rarity"] == "Secret Rare"
    assert body["artworks"] and body["artworks"][0]["id"] == ARMADES_ID


def test_get_unknown_card_is_404(client: TestClient) -> None:
    response = client.get("/api/cards/1")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_search_finds_armades(client: TestClient) -> None:
    response = client.get("/api/search", params={"q": "arm"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["query"] == "arm"
    assert [c["id"] for c in body["results"]] == [ARMADES_ID]
    assert body["results"][0]["name"] == "Armades, Keeper of Boundaries"

    limited = client.get("/api/search", params={"q": "dark magician", "limit": 1})
    assert limited.status_code == 200 and len(limited.json()["results"]) == 1


def test_search_rejects_short_or_missing_query(client: TestClient) -> None:
    assert client.get("/api/search", params={"q": "a"}).status_code == 400
    assert client.get("/api/search", params={"q": " x "}).status_code == 400
    assert client.get("/api/search").status_code == 422


# --------------------------------------------------------------- recognize
def test_recognize_rejects_text_plain(client: TestClient) -> None:
    response = upload(client, b"hello world, this is not an image", "text/plain")
    assert response.status_code == 415, response.text
    assert "not a recognizable image" in response.json()["detail"]


def test_recognize_rejects_oversize_upload(client: TestClient) -> None:
    payload = b"\x89PNG\r\n\x1a\n" + bytes(UPLOAD_LIMIT + 100_000)
    response = upload(client, payload, "image/png")
    assert response.status_code == 413, response.text
    assert "limit" in response.json()["detail"]


def test_recognize_rejects_undecodable_image_bytes(client: TestClient) -> None:
    response = upload(client, b"\x89PNG\r\n\x1a\n" + bytes(200), "image/png")
    assert response.status_code == 422, response.text
    assert "could not decode" in response.json()["detail"]


def test_recognize_requires_the_image_field(client: TestClient) -> None:
    response = upload(client, jpeg_bytes(), "image/jpeg", field="photo")
    assert response.status_code == 422
    assert "image" in response.json()["detail"]

    not_multipart = client.post("/api/recognize", content=b"{}", headers={"content-type": "application/json"})
    assert not_multipart.status_code == 415


def test_recognize_accepts_wrong_content_type_when_bytes_are_an_image(client: TestClient) -> None:
    response = upload(client, jpeg_bytes(), "application/octet-stream")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == RecognitionStatus.CARD_NOT_DETECTED.value  # a flat square has no card-shaped contour
    assert body["detection"]["detected"] is False


def test_recognize_synthetic_card_returns_valid_result(client: TestClient) -> None:
    response = upload(client, png_bytes(synthetic_scene()), "image/png", query="?debug=true")
    assert response.status_code == 200, response.text
    body = response.json()
    result = RecognitionResult.model_validate(body)
    assert result.status in RecognitionStatus
    assert result.status is RecognitionStatus.OCR_FAILED  # fake OCR reads nothing
    assert result.detection is not None and result.detection.detected
    assert result.detection.method not in (None, "full_image")
    assert {"detection", "ocr", "resolve", "total"} <= set(result.timing_ms)
    assert result.debug is None  # debug images are off unless SAVE_DEBUG_IMAGES / DEBUG is enabled


def test_recognize_writes_debug_images_when_enabled(api_settings: Settings) -> None:
    api_settings.save_debug_images = True
    with TestClient(create_app(api_settings)) as client:
        response = upload(client, png_bytes(synthetic_scene()), "image/png", query="?debug=true")
        assert response.status_code == 200, response.text
        debug = response.json()["debug"]
        assert debug and "01_detected_contour" in debug and "04_name_roi_0" in debug
        assert all(Path(p).is_file() for p in debug.values())
        assert all(Path(p).is_relative_to(api_settings.resolved_debug_dir) for p in debug.values())

        without_flag = upload(client, png_bytes(synthetic_scene()), "image/png")
        assert without_flag.status_code == 200 and without_flag.json()["debug"] is None


def test_unexpected_error_is_a_json_500(api_settings: Settings) -> None:
    app = create_app(api_settings)
    with TestClient(app, raise_server_exceptions=False) as client:

        def explode(*_args, **_kwargs):
            raise RuntimeError("boom")

        app.state.service.recognize = explode  # type: ignore[method-assign]
        response = upload(client, jpeg_bytes(), "image/jpeg")
        assert response.status_code == 500
        assert response.json() == {"detail": "internal error"}
