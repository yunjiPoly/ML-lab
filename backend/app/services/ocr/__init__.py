"""OCR providers.  ``OCRProvider`` is the only thing the pipeline depends on.

Concrete providers live in :mod:`paddle_provider` (real engine, lazily loaded)
and :mod:`fake_provider` (tests); :func:`build_ocr_provider` picks one from
settings.  Importing this package never imports Paddle.
"""

from app.services.ocr.base import DebugSink, OCRAlternative, OCRProvider, OCRResult
from app.services.ocr.factory import build_ocr_provider
from app.services.ocr.fake_provider import FakeOCRProvider
from app.services.ocr.paddle_provider import PaddleOCRProvider

__all__ = [
    "DebugSink",
    "FakeOCRProvider",
    "OCRAlternative",
    "OCRProvider",
    "OCRResult",
    "PaddleOCRProvider",
    "build_ocr_provider",
]
