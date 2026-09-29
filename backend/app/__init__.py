"""Yu-Gi-Oh! card recognition backend package.

Architecture (see README):

    photo -> card_detection (OpenCV geometry) -> orientation -> ROI crops
          -> ocr (PaddleOCR text recognition on crops) -> text normalization
          -> resolver (database constraints + fuzzy matching) -> RecognitionResult

Optional visual recognition (DINOv2 etc.) is a pluggable fallback behind
``app.services.visual.VisualRecognizer`` and is disabled by default.
"""

__version__ = "0.1.0"
