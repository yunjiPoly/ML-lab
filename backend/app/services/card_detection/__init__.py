"""Card detection, perspective correction, orientation and ROI extraction (OpenCV only)."""

from app.services.card_detection.debug import DebugImageWriter, draw_detection, draw_rois
from app.services.card_detection.detector import CardDetector, order_points
from app.services.card_detection.orientation import (
    OrientationCorrector,
    TextDensityOrientationCorrector,
    rotate_image,
)
from app.services.card_detection.roi import RoiCrop, extract_roi_crops
from app.services.card_detection.service import CardGeometryService
from app.services.card_detection.types import CardDetection, CardNotDetectedError, NormalizedCard
from app.services.card_detection.warp import normalize_card, warp_card

__all__ = [
    "CardDetection",
    "CardDetector",
    "CardGeometryService",
    "CardNotDetectedError",
    "DebugImageWriter",
    "NormalizedCard",
    "OrientationCorrector",
    "RoiCrop",
    "TextDensityOrientationCorrector",
    "draw_detection",
    "draw_rois",
    "extract_roi_crops",
    "normalize_card",
    "order_points",
    "rotate_image",
    "warp_card",
]
