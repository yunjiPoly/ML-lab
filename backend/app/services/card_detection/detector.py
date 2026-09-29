"""Card quadrilateral detection with classic OpenCV (no machine learning).

The detector works on a downscaled copy of the photo, builds several binary
edge / mask images with different strategies (Canny at a few thresholds, Canny
plus morphological closing, adaptive thresholds in both polarities and the HSV
saturation / value channels for sleeved cards on busy playmats), extracts
convex four-point contours from each of them and scores every candidate on its
size, rectangularity and closeness to the physical card aspect ratio
(59 mm x 86 mm, ~0.686).  The corners of a candidate are refined by fitting
straight lines to the contour sides (which neutralises the rounded card corners
and the pixel jaggies of a rotated card) and finally scaled back to source
pixels.

Nothing here invents a card: when no plausible quadrilateral exists the
detector either returns a low-score ``full_image`` detection (only when the
whole image already has card proportions and the fallback is enabled) or
``None``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import cv2
import numpy as np

from app.core.layout import CARD_ASPECT_MAX, CARD_ASPECT_MIN, CARD_ASPECT_RATIO
from app.services.card_detection.types import CardDetection

logger = logging.getLogger(__name__)

DEFAULT_WORKING_LONG_SIDE: int = 1000
MAX_AREA_RATIO: float = 0.95  # larger quads are just the image frame (handled by the fallback)
MIN_RECTANGULARITY: float = 0.80
MIN_CORNER_ANGLE_DEG: float = 50.0
MIN_SIDE_PX: float = 10.0
EARLY_STOP_SCORE: float = 0.92
FULL_IMAGE_SCORE: float = 0.20
APPROX_EPSILONS: tuple[float, ...] = (0.02, 0.035, 0.05)
FULL_IMAGE_METHOD: str = "full_image"

MaskIterator = Iterator[tuple[str, np.ndarray]]


# --------------------------------------------------------------------------- geometry helpers
def order_points(points: np.ndarray | list) -> np.ndarray:
    """Order four points clockwise starting at the top-left: TL, TR, BR, BL.

    The points are sorted by their angle around the centroid (clockwise in
    image coordinates, where y grows downwards) and the cycle is rotated so
    that the corner with the smallest ``x + y`` comes first.  Unlike the
    classic sum/difference trick this never assigns one point to two slots.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if pts.shape != (4, 2):
        raise ValueError(f"order_points expects 4 points, got shape {pts.shape}")
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]
    start = int(np.argmin(pts.sum(axis=1)))
    return np.roll(pts, -start, axis=0).astype(np.float32)


def quad_dimensions(corners: np.ndarray) -> tuple[float, float]:
    """Return the mean (width, height) of an ordered TL, TR, BR, BL quadrilateral."""
    tl, tr, br, bl = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
    height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
    return float(width), float(height)


def _min_corner_angle(corners: np.ndarray) -> float:
    """Smallest interior angle (degrees) of a quadrilateral."""
    smallest = 180.0
    for i in range(4):
        prev_pt, pt, next_pt = corners[i - 1], corners[i], corners[(i + 1) % 4]
        a, b = prev_pt - pt, next_pt - pt
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom < 1e-9:
            return 0.0
        cos = float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))
        smallest = min(smallest, float(np.degrees(np.arccos(cos))))
    return smallest


def _intersect(p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray) -> np.ndarray | None:
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-9:
        return None
    diff = p2 - p1
    s = (diff[0] * d2[1] - diff[1] * d2[0]) / cross
    return p1 + s * d1


def refine_corners(contour_points: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Refine an approximate quad by fitting a line to the contour points of each side.

    Points close to a side (within 2 % of its length, at least 2 px) and away
    from the corners (the outer 8 % is excluded so the rounded card corners do
    not bias the fit) are fitted with a robust Huber line; consecutive lines
    are intersected to obtain the new corners.  The original corners are
    returned when a side has too few supporting points or the refinement would
    move a corner implausibly far.
    """
    pts = np.asarray(contour_points, dtype=np.float32).reshape(-1, 2)
    corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    lines: list[tuple[np.ndarray, np.ndarray]] = []
    max_side = 0.0
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        direction = b - a
        length = float(np.linalg.norm(direction))
        if length < 1e-6:
            return corners
        max_side = max(max_side, length)
        unit = direction / length
        normal = np.array([-unit[1], unit[0]], dtype=np.float32)
        rel = pts - a
        along = rel @ unit
        dist = np.abs(rel @ normal)
        mask = (dist <= max(2.0, 0.02 * length)) & (along >= 0.08 * length) & (along <= 0.92 * length)
        support = pts[mask]
        if len(support) < 8:
            lines.append((a, unit))
            continue
        vx, vy, x0, y0 = cv2.fitLine(support, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0], dtype=np.float32), np.array([vx, vy], dtype=np.float32)))
    refined = []
    for i in range(4):
        (p_prev, d_prev), (p_cur, d_cur) = lines[i - 1], lines[i]
        point = _intersect(p_prev, d_prev, p_cur, d_cur)
        if point is None:
            return corners
        refined.append(point)
    result = np.asarray(refined, dtype=np.float32)
    if np.any(np.linalg.norm(result - corners, axis=1) > 0.05 * max_side):
        return corners
    return result


# ------------------------------------------------------------------------ working image
@dataclass
class _WorkingImage:
    """Downscaled BGR image plus the derived channels the strategies share."""

    bgr: np.ndarray
    scale: float  # working pixels per source pixel
    gray: np.ndarray
    blurred: np.ndarray

    @property
    def area(self) -> int:
        h, w = self.gray.shape[:2]
        return int(h * w)


def _prepare(image: np.ndarray, working_long_side: int) -> _WorkingImage:
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        bgr = image
    h, w = bgr.shape[:2]
    scale = 1.0
    longest = max(h, w)
    if longest > working_long_side:
        scale = working_long_side / float(longest)
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return _WorkingImage(bgr=bgr, scale=scale, gray=gray, blurred=blurred)


# ---------------------------------------------------------------------------- strategies
def _canny_masks(work: _WorkingImage) -> MaskIterator:
    kernel = np.ones((3, 3), np.uint8)
    for low, high in ((50, 150), (30, 90), (80, 200)):
        edges = cv2.Canny(work.blurred, low, high)
        yield f"canny_{low}_{high}", cv2.dilate(edges, kernel, iterations=1)


def _closed_canny_masks(work: _WorkingImage) -> MaskIterator:
    edges = cv2.Canny(work.blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    yield "canny_close", cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)


def _adaptive_masks(work: _WorkingImage) -> MaskIterator:
    block = max(31, (min(work.gray.shape[:2]) // 20) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for name, mode in (("adaptive_light", cv2.THRESH_BINARY), ("adaptive_dark", cv2.THRESH_BINARY_INV)):
        mask = cv2.adaptiveThreshold(work.blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, mode, block, 5)
        yield name, cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def _hsv_masks(work: _WorkingImage) -> MaskIterator:
    hsv = cv2.cvtColor(work.bgr, cv2.COLOR_BGR2HSV)
    saturation = cv2.GaussianBlur(hsv[:, :, 1], (5, 5), 0)
    value = cv2.GaussianBlur(hsv[:, :, 2], (5, 5), 0)
    kernel3 = np.ones((3, 3), np.uint8)
    yield "canny_saturation", cv2.dilate(cv2.Canny(saturation, 40, 120), kernel3, iterations=1)
    yield "canny_value", cv2.dilate(cv2.Canny(value, 50, 150), kernel3, iterations=1)
    _, otsu = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    kernel7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel7)
    yield "saturation_otsu", cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel7)


STRATEGIES: tuple[Callable[[_WorkingImage], MaskIterator], ...] = (
    _canny_masks,
    _closed_canny_masks,
    _adaptive_masks,
    _hsv_masks,
)


# ---------------------------------------------------------------------------- candidates
@dataclass(frozen=True)
class _Candidate:
    corners: np.ndarray  # working-image pixels, ordered TL, TR, BR, BL
    method: str
    score: float
    aspect: float
    rectangularity: float
    area_ratio: float


def _approximate_quad(contour: np.ndarray) -> np.ndarray | None:
    """Reduce a contour to a convex quadrilateral, or ``None`` when it is not one."""
    perimeter = cv2.arcLength(contour, True)
    for eps in APPROX_EPSILONS:
        approx = cv2.approxPolyDP(contour, eps * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float32)
    hull = cv2.convexHull(contour)
    if cv2.contourArea(hull) > 1.15 * max(cv2.contourArea(contour), 1.0):
        return None
    perimeter = cv2.arcLength(hull, True)
    for eps in APPROX_EPSILONS:
        approx = cv2.approxPolyDP(hull, eps * perimeter, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    return None


def score_quad(aspect: float, rectangularity: float, area_ratio: float) -> float:
    """Combine geometric plausibility measures into a score in [0, 1]."""
    aspect_score = 1.0 - min(1.0, abs(aspect - CARD_ASPECT_RATIO) / (CARD_ASPECT_RATIO - CARD_ASPECT_MIN))
    area_score = min(1.0, area_ratio / 0.25)
    rect_score = float(np.clip(rectangularity, 0.0, 1.0))
    return round(0.45 * aspect_score + 0.35 * rect_score + 0.20 * area_score, 4)


class CardDetector:
    """Locate the card quadrilateral in a photo.

    Args:
        min_area_ratio: minimum contour area relative to the image area.
        aspect_min / aspect_max: accepted short/long side ratio (either orientation).
        allow_full_image_fallback: treat the whole image as the card when no
            contour is found but the image itself has card proportions.
        working_long_side: the photo is downscaled to this longest side for detection.
    """

    def __init__(
        self,
        *,
        min_area_ratio: float = 0.05,
        aspect_min: float = CARD_ASPECT_MIN,
        aspect_max: float = CARD_ASPECT_MAX,
        allow_full_image_fallback: bool = True,
        working_long_side: int = DEFAULT_WORKING_LONG_SIDE,
    ) -> None:
        self._min_area_ratio = min_area_ratio
        self._aspect_min = aspect_min
        self._aspect_max = aspect_max
        self._allow_full_image_fallback = allow_full_image_fallback
        self._working_long_side = working_long_side

    # ----------------------------------------------------------------- public
    def detect(self, image: np.ndarray) -> CardDetection | None:
        """Return the best card quadrilateral (source pixels) or ``None``."""
        if image is None or image.size == 0 or image.ndim not in (2, 3):
            raise ValueError("detect() expects a non-empty HxW or HxWxC image array")
        work = _prepare(image, self._working_long_side)
        min_area = self._min_area_ratio * work.area
        best: _Candidate | None = None
        candidates = 0
        for strategy in STRATEGIES:
            for method, mask in strategy(work):
                for candidate in self._quads_from_mask(mask, method, work, min_area):
                    candidates += 1
                    if best is None or candidate.score > best.score:
                        best = candidate
            if best is not None and best.score >= EARLY_STOP_SCORE:
                break
        if best is None:
            logger.info("No card contour found (%d candidates rejected)", candidates)
            return self._fallback(image)
        corners = (best.corners / work.scale).astype(np.float32)
        logger.info(
            "Card detected via %s score=%.3f aspect=%.3f rect=%.3f area=%.3f (%d candidates)",
            best.method, best.score, best.aspect, best.rectangularity, best.area_ratio, candidates,
        )
        return CardDetection(
            corners=corners,
            method=best.method,
            score=best.score,
            details={
                "aspect": round(best.aspect, 4),
                "rectangularity": round(best.rectangularity, 4),
                "area_ratio": round(best.area_ratio, 4),
                "candidates": candidates,
                "working_scale": round(work.scale, 4),
            },
        )

    # --------------------------------------------------------------- internals
    def _quads_from_mask(
        self, mask: np.ndarray, method: str, work: _WorkingImage, min_area: float
    ) -> Iterator[_Candidate]:
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > MAX_AREA_RATIO * work.area:
                continue
            quad = _approximate_quad(contour)
            if quad is None:
                continue
            candidate = self._evaluate(order_points(quad), contour, method, work.area)
            if candidate is not None:
                yield candidate

    def _evaluate(
        self, corners: np.ndarray, contour: np.ndarray, method: str, image_area: int
    ) -> _Candidate | None:
        width, height = quad_dimensions(corners)
        if min(width, height) < MIN_SIDE_PX:
            return None
        aspect = min(width, height) / max(width, height)
        if not self._aspect_min <= aspect <= self._aspect_max:
            return None
        if _min_corner_angle(corners) < MIN_CORNER_ANGLE_DEG:
            return None
        quad_area = float(cv2.contourArea(corners))
        (_, (rw, rh), _) = cv2.minAreaRect(corners)
        rect_area = float(rw * rh)
        rectangularity = quad_area / rect_area if rect_area > 0 else 0.0
        if rectangularity < MIN_RECTANGULARITY:
            return None
        area_ratio = quad_area / float(image_area)
        if area_ratio > MAX_AREA_RATIO:
            return None
        refined = refine_corners(contour.reshape(-1, 2), corners)
        return _Candidate(
            corners=refined,
            method=method,
            score=score_quad(aspect, rectangularity, area_ratio),
            aspect=aspect,
            rectangularity=rectangularity,
            area_ratio=area_ratio,
        )

    def _fallback(self, image: np.ndarray) -> CardDetection | None:
        if not self._allow_full_image_fallback:
            return None
        h, w = image.shape[:2]
        aspect = min(w, h) / max(w, h)
        if not self._aspect_min <= aspect <= self._aspect_max:
            logger.info(
                "Full-image fallback rejected: image aspect %.3f outside [%.2f, %.2f]",
                aspect, self._aspect_min, self._aspect_max,
            )
            return None
        logger.info("Using the whole %dx%d image as the card (aspect %.3f)", w, h, aspect)
        corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        return CardDetection(
            corners=corners,
            method=FULL_IMAGE_METHOD,
            score=FULL_IMAGE_SCORE,
            details={"aspect": round(aspect, 4), "reason": "no card contour found"},
        )
