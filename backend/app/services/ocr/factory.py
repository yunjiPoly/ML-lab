"""Build the configured :class:`OCRProvider` from :class:`Settings`."""

from __future__ import annotations

import logging

from app.core.config import Settings
from app.services.ocr.base import OCRProvider
from app.services.ocr.fake_provider import FakeOCRProvider
from app.services.ocr.paddle_provider import PaddleOCRProvider

logger = logging.getLogger(__name__)

KNOWN_PROVIDERS: tuple[str, ...] = ("paddle", "fake")


def build_ocr_provider(settings: Settings) -> OCRProvider:
    """Return the provider named by ``settings.ocr_provider``.

    ``"paddle"`` -> :class:`PaddleOCRProvider` (model loaded lazily on first
    use), ``"fake"`` -> :class:`FakeOCRProvider` with empty readings.
    Raises :class:`ValueError` for anything else.
    """
    key = (settings.ocr_provider or "").strip().lower()
    if key == "paddle":
        logger.info("OCR provider: paddle (model=%s, device=%s)", settings.ocr_rec_model, settings.ocr_device)
        return PaddleOCRProvider(model_name=settings.ocr_rec_model, device=settings.ocr_device)
    if key == "fake":
        logger.info("OCR provider: fake")
        return FakeOCRProvider()
    raise ValueError(f"Unknown OCR provider {settings.ocr_provider!r}; expected one of {KNOWN_PROVIDERS}")
