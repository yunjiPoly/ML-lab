from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VisualCandidate:
    """A card suggested by visual similarity."""

    card_id: int
    score: float
    artwork_id: int | None = None
    reason: str = "visual_similarity"


class VisualRecognizer(ABC):
    """Image-embedding based recognizer (e.g. DINOv2 + nearest-neighbour search)."""

    name: str = "base"

    @property
    def available(self) -> bool:
        """False when the implementation cannot run (model missing, disabled...)."""
        return True

    @abstractmethod
    def recognize(self, image: np.ndarray) -> list[VisualCandidate]:
        """Return ranked candidates for a normalized (upright) card image."""


class NoOpVisualRecognizer(VisualRecognizer):
    """Default: visual recognition disabled.  Keeps the 300+ MB vision model optional."""

    name = "noop"

    @property
    def available(self) -> bool:
        return False

    def recognize(self, image: np.ndarray) -> list[VisualCandidate]:
        return []


def build_visual_recognizer(kind: str = "noop") -> VisualRecognizer:
    """Factory used by the pipeline.  Unknown kinds fall back to no-op with a warning."""
    if kind in ("", "noop", "none", "disabled"):
        return NoOpVisualRecognizer()
    # Future: if kind == "dinov2": from app.services.visual.dinov2 import DinoV2Recognizer ...
    import logging

    logging.getLogger(__name__).warning("Unknown visual recognizer '%s'; using noop.", kind)
    return NoOpVisualRecognizer()
