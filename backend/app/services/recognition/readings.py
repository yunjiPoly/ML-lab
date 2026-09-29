"""OCR reading helpers shared by the recognition pipeline.

Small, side-effect-free pieces: plausibility checks for set-code readings,
the container for one OCR pass (:class:`OcrReadings`), a debug-sink wrapper
that prefixes keys, and a fallback conversion to :class:`OcrFieldResult`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.text import looks_like_set_code, normalize_card_name, normalize_set_code, set_code_variants
from app.schemas.recognition import OcrFieldResult
from app.services.ocr.base import DebugSink, OCRResult


def reading_is_set_code(result: OCRResult | None) -> bool:
    """True when the *selected* reading has (or can be coerced into) a set-code shape."""
    if result is None or result.is_empty:
        return False
    return looks_like_set_code(result.text) or bool(set_code_variants(result.text))


def has_plausible_set_code(result: OCRResult | None) -> bool:
    """True when the selected reading or any alternative looks like a set code."""
    if result is None or result.is_empty:
        return False
    return any(looks_like_set_code(text) or bool(set_code_variants(text)) for text, _ in result.all_readings())


class PrefixedDebugSink:
    """Debug sink wrapper that prefixes every key (keeps retry images from overwriting the first pass)."""

    def __init__(self, sink: DebugSink, prefix: str) -> None:
        self._sink = sink
        self._prefix = prefix

    def save(self, key: str, image: np.ndarray) -> Path:
        return self._sink.save(f"{self._prefix}{key}", image)


@dataclass
class OcrReadings:
    """Name and set-code readings from one OCR pass over a normalized card."""

    name: OCRResult | None
    set_code: OCRResult | None
    name_roi: str | None = None
    set_code_roi: str | None = None
    extra_rotation: int = 0

    @property
    def name_confidence(self) -> float:
        return 0.0 if self.name is None or self.name.is_empty else self.name.confidence

    @property
    def set_code_confidence(self) -> float:
        return 0.0 if self.set_code is None or self.set_code.is_empty else self.set_code.confidence

    @property
    def plausible_set_code(self) -> bool:
        return has_plausible_set_code(self.set_code)

    def is_weak(self, min_confidence: float) -> bool:
        """Both fields unusable: name below ``min_confidence`` and no set-code-shaped reading."""
        return self.name_confidence < min_confidence and not self.plausible_set_code

    def strength(self) -> float:
        """Comparable score of an attempt: a plausible set code dominates, then OCR confidences."""
        return (2.0 if self.plausible_set_code else 0.0) + self.set_code_confidence + self.name_confidence


def field_from_ocr(result: OCRResult | None, *, kind: str) -> OcrFieldResult | None:
    """Fallback :class:`OcrFieldResult` when the resolver could not produce one."""
    if result is None:
        return None
    normalize = normalize_set_code if kind == "set_code" else normalize_card_name
    return OcrFieldResult(
        raw=result.text,
        normalized=normalize(result.text),
        confidence=max(0.0, min(1.0, result.confidence)),
        variant=result.variant,
        alternatives=[alt.text for alt in result.alternatives if alt.text.strip()],
    )
