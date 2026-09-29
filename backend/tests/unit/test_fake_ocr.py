"""FakeOCRProvider behaviour and the provider factory (no Paddle involved)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from app.core.config import Settings
from app.services.ocr import FakeOCRProvider, OCRProvider, PaddleOCRProvider, build_ocr_provider


class RecordingSink:
    def __init__(self) -> None:
        self.saved: dict[str, np.ndarray] = {}

    def save(self, key: str, image: np.ndarray) -> Path:
        self.saved[key] = image
        return Path(f"{key}.png")


@pytest.fixture()
def crop() -> np.ndarray:
    return np.full((30, 100, 3), 200, dtype=np.uint8)


def test_fake_provider_returns_configured_readings(crop: np.ndarray) -> None:
    provider = FakeOCRProvider(
        name_text="Armades, Keeper of Boundaries", name_confidence=0.9, set_code_text="JOTL-EN045", set_code_confidence=0.99
    )
    assert isinstance(provider, OCRProvider) and provider.name == "fake"
    name = provider.recognize_name(crop)
    assert (name.text, name.confidence, name.variant) == ("Armades, Keeper of Boundaries", 0.9, "fake")
    code = provider.recognize_set_code(crop)
    assert (code.text, code.confidence) == ("JOTL-EN045", 0.99)
    text = provider.recognize_text(crop)
    assert text.text == "Armades, Keeper of Boundaries"
    assert provider.calls == ["name", "set_code", "text"]
    provider.warmup()  # no-op


def test_fake_provider_empty_text_gives_empty_result(crop: np.ndarray) -> None:
    provider = FakeOCRProvider()
    assert provider.recognize_name(crop).is_empty
    assert provider.recognize_set_code(crop).is_empty
    assert provider.recognize_text(crop).is_empty
    assert provider.recognize_name(crop).confidence == 0.0


def test_fake_provider_saves_debug_images(crop: np.ndarray) -> None:
    provider = FakeOCRProvider(name_text="x", name_confidence=1.0)
    sink = RecordingSink()
    provider.recognize_name(crop, debug=sink)
    provider.recognize_set_code(crop, debug=sink)
    assert set(sink.saved) == {"ocr_name_raw", "ocr_set_code_raw"}


# ------------------------------------------------------------ factory
def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def test_factory_builds_fake(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OCR_PROVIDER", raising=False)
    provider = build_ocr_provider(_settings(ocr_provider="fake"))
    assert isinstance(provider, FakeOCRProvider)


def test_factory_builds_paddle_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OCR_PROVIDER", raising=False)
    provider = build_ocr_provider(_settings(ocr_provider="Paddle", ocr_rec_model="PP-OCRv6_small_rec", ocr_device="gpu"))
    assert isinstance(provider, PaddleOCRProvider)
    assert provider.name == "paddle"
    assert provider.model_name == "PP-OCRv6_small_rec" and provider.device == "gpu"
    assert provider.is_loaded is False
    assert "paddleocr" not in sys.modules  # construction must not import the engine


def test_factory_rejects_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OCR_PROVIDER", raising=False)
    with pytest.raises(ValueError, match="Unknown OCR provider"):
        build_ocr_provider(_settings(ocr_provider="tesseract"))
