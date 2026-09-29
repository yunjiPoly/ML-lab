"""Debug image persistence and overlay drawing helpers."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import cv2
import numpy as np

from app.core.layout import CardLayoutTemplate

logger = logging.getLogger(__name__)

_SAFE_KEY = re.compile(r"[^A-Za-z0-9._-]+")
NAME_COLOR: tuple[int, int, int] = (0, 255, 0)  # green (BGR)
SET_CODE_COLOR: tuple[int, int, int] = (255, 0, 0)  # blue (BGR)
POLYGON_COLOR: tuple[int, int, int] = (0, 255, 0)
LABEL_COLOR: tuple[int, int, int] = (0, 255, 255)  # yellow


def _to_uint8_image(image: np.ndarray) -> np.ndarray:
    """Coerce masks / float images into something ``imencode`` accepts."""
    if image.dtype == np.bool_:
        return image.astype(np.uint8) * 255
    if image.dtype != np.uint8:
        array = image.astype(np.float32)
        if array.size and float(array.max()) <= 1.0:
            array *= 255.0
        return np.clip(array, 0, 255).astype(np.uint8)
    return image


class DebugImageWriter:
    """Write intermediate images as ``<directory>/<prefix><key>.jpg`` and remember them.

    Satisfies the ``DebugSink`` protocol of the OCR layer.  Files are encoded
    in memory and written with :meth:`Path.write_bytes` so non-ASCII paths
    work on Windows (``cv2.imwrite`` does not).
    """

    def __init__(self, directory: Path, *, prefix: str = "") -> None:
        self.directory = Path(directory)
        self.prefix = prefix
        self._paths: dict[str, Path] = {}

    def save(self, key: str, image: np.ndarray) -> Path:
        """Persist ``image`` under ``key`` and return the written path."""
        if image is None or image.size == 0:
            raise ValueError(f"Refusing to save empty debug image '{key}'")
        self.directory.mkdir(parents=True, exist_ok=True)
        safe_key = _SAFE_KEY.sub("_", key).strip("._") or "image"
        path = self.directory / f"{self.prefix}{safe_key}.jpg"
        ok, buffer = cv2.imencode(".jpg", _to_uint8_image(image), [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise RuntimeError(f"Could not encode debug image '{key}'")
        path.write_bytes(buffer.tobytes())
        self._paths[key] = path
        logger.debug("Saved debug image %s -> %s", key, path)
        return path

    def paths(self) -> dict[str, str]:
        """Mapping of saved key -> absolute file path (as strings)."""
        return {key: str(path) for key, path in self._paths.items()}


def _line_thickness(image: np.ndarray) -> int:
    return max(2, int(round(min(image.shape[:2]) / 400)))


def draw_detection(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Return a copy of ``image`` with the card polygon and TL/TR/BR/BL labels drawn."""
    canvas = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    pts = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    thickness = _line_thickness(canvas)
    cv2.polylines(canvas, [np.round(pts).astype(np.int32)], True, POLYGON_COLOR, thickness)
    font_scale = max(0.5, min(canvas.shape[:2]) / 900)
    for label, (x, y) in zip(("TL", "TR", "BR", "BL"), pts, strict=True):
        center = (int(round(x)), int(round(y)))
        cv2.circle(canvas, center, thickness * 3, LABEL_COLOR, -1)
        origin = (center[0] + thickness * 4, center[1] - thickness * 4)
        cv2.putText(canvas, label, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2)
        cv2.putText(canvas, label, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, LABEL_COLOR, thickness)
    return canvas


def draw_rois(card: np.ndarray, layout: CardLayoutTemplate) -> np.ndarray:
    """Return a copy of the canonical card with name ROIs (green) and set-code ROIs (blue)."""
    canvas = card.copy() if card.ndim == 3 else cv2.cvtColor(card, cv2.COLOR_GRAY2BGR)
    h, w = canvas.shape[:2]
    factor = w / layout.width if layout.width else 1.0
    scaled = layout.scaled(factor) if abs(factor - 1.0) > 1e-6 else layout
    for rois, color in ((scaled.name_rois, NAME_COLOR), (scaled.set_code_rois, SET_CODE_COLOR)):
        for roi in rois:
            r = roi.clamp(w, h)
            cv2.rectangle(canvas, (r.x1, r.y1), (r.x2 - 1, r.y2 - 1), color, 2)
            cv2.putText(canvas, roi.key, (r.x1 + 2, max(10, r.y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    return canvas
