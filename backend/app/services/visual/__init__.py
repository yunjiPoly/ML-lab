"""Optional visual (image-embedding) recognition fallback.

Disabled by default (``NoOpVisualRecognizer``).  A DINOv2 + FAISS/NumPy
implementation can be added later by implementing ``VisualRecognizer`` and
registering it in ``build_visual_recognizer`` without touching the pipeline.
"""

from app.services.visual.base import NoOpVisualRecognizer, VisualCandidate, VisualRecognizer, build_visual_recognizer

__all__ = ["NoOpVisualRecognizer", "VisualCandidate", "VisualRecognizer", "build_visual_recognizer"]
