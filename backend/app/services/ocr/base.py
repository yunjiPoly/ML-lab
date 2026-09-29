"""OCR provider interface.

Implementations (PaddleOCR today; Google Cloud Vision / Azure later) receive an
already-cropped BGR ``numpy`` image of a single text region and return an
:class:`OCRResult`.  Preprocessing-variant selection happens inside the provider
so the resolver never has to know which engine produced the text.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class DebugSink(Protocol):
    """Anything that can persist a debug image (satisfied by ``DebugImageWriter``)."""

    def save(self, key: str, image: np.ndarray) -> Path: ...


@dataclass(frozen=True)
class OCRAlternative:
    """A secondary reading (another preprocessing variant) that was not selected."""

    text: str
    confidence: float
    variant: str | None = None


@dataclass
class OCRResult:
    """Result of recognizing one text region."""

    text: str
    confidence: float
    raw_result: Any = None
    variant: str | None = None
    alternatives: list[OCRAlternative] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    @classmethod
    def empty(cls, raw_result: Any = None) -> "OCRResult":
        return cls(text="", confidence=0.0, raw_result=raw_result)

    def all_readings(self) -> list[tuple[str, float]]:
        """Selected reading first, then alternatives, de-duplicated, best score kept."""
        seen: dict[str, float] = {}
        for text, score in [(self.text, self.confidence), *((a.text, a.confidence) for a in self.alternatives)]:
            text = text.strip()
            if not text:
                continue
            if text not in seen or score > seen[text]:
                seen[text] = score
        return sorted(seen.items(), key=lambda kv: kv[1], reverse=True)


class OCRProvider(ABC):
    """Pluggable OCR engine."""

    name: str = "base"

    def warmup(self) -> None:
        """Load models eagerly (optional)."""

    @abstractmethod
    def recognize_name(self, image: np.ndarray, *, debug: DebugSink | None = None) -> OCRResult:
        """Recognize a card-name crop (BGR uint8 array).

        When ``debug`` is given, implementations should save every preprocessed
        variant they tried (``debug.save("ocr_name_<variant>", img)``).
        """

    @abstractmethod
    def recognize_set_code(self, image: np.ndarray, *, debug: DebugSink | None = None) -> OCRResult:
        """Recognize a set-code crop (BGR uint8 array).  See :meth:`recognize_name` for ``debug``."""

    @abstractmethod
    def recognize_text(self, image: np.ndarray) -> OCRResult:
        """Recognize a single text line without any preprocessing variants (debug helper)."""
