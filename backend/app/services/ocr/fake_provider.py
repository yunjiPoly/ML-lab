"""Deterministic OCR provider for tests and offline development."""

from __future__ import annotations

import numpy as np

from app.services.ocr.base import DebugSink, OCRProvider, OCRResult


class FakeOCRProvider(OCRProvider):
    """Return pre-configured readings regardless of the image content.

    ``recognize_name`` / ``recognize_text`` yield ``name_text``,
    ``recognize_set_code`` yields ``set_code_text``.  Empty configured text
    gives :meth:`OCRResult.empty` so the pipeline's "nothing read" branches can
    be exercised too.  When a ``debug`` sink is passed the raw crop is saved
    under the same keys a real provider would use.
    """

    name = "fake"

    def __init__(
        self,
        *,
        name_text: str = "",
        name_confidence: float = 0.0,
        set_code_text: str = "",
        set_code_confidence: float = 0.0,
    ) -> None:
        self.name_text = name_text
        self.name_confidence = float(name_confidence)
        self.set_code_text = set_code_text
        self.set_code_confidence = float(set_code_confidence)
        self.calls: list[str] = []

    @staticmethod
    def _make(text: str, confidence: float, variant: str) -> OCRResult:
        if not text.strip():
            return OCRResult.empty(raw_result={"provider": "fake"})
        return OCRResult(text=text, confidence=confidence, raw_result={"provider": "fake"}, variant=variant)

    def recognize_name(self, image: np.ndarray, *, debug: DebugSink | None = None) -> OCRResult:
        self.calls.append("name")
        if debug is not None and image is not None and image.size:
            debug.save("ocr_name_raw", image)
        return self._make(self.name_text, self.name_confidence, "fake")

    def recognize_set_code(self, image: np.ndarray, *, debug: DebugSink | None = None) -> OCRResult:
        self.calls.append("set_code")
        if debug is not None and image is not None and image.size:
            debug.save("ocr_set_code_raw", image)
        return self._make(self.set_code_text, self.set_code_confidence, "fake")

    def recognize_text(self, image: np.ndarray) -> OCRResult:
        self.calls.append("text")
        return self._make(self.name_text, self.name_confidence, "fake")
