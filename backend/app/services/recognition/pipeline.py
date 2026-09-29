"""End-to-end recognition pipeline.

``photo -> CardGeometryService -> ROI crops -> OCRProvider -> Resolver
(-> VisualRecognizer fallback) -> RecognitionResult``

The service never raises for a bad photo: geometry failures become
``CARD_NOT_DETECTED`` and unexpected OCR / resolver errors become
``OCR_FAILED`` with an explanatory note, so the API can answer 200 with an
honest status instead of a 500.

OCR strategy
------------
Every name ROI of the layout is read in template order and the most confident
non-empty reading is kept (stopping early at ``EARLY_STOP_CONFIDENCE``).  Set
code ROIs are read the same way but readings that have a set-code shape
(:func:`looks_like_set_code` / :func:`set_code_variants`) are preferred over
more confident readings that do not.  When both fields are weak (name below
``ocr_min_confidence`` and no plausible set code) the card is rotated 180
degrees and read again, keeping the better attempt: this catches the case
where the orientation heuristic guessed wrong.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from app.core.config import Settings
from app.core.layout import CardLayoutTemplate, get_layout
from app.db.protocols import CardRepository
from app.schemas.recognition import (
    DetectionInfo,
    RecognitionOcr,
    RecognitionResult,
    RecognitionStatus,
    ResolutionResult,
)
from app.services.card_detection.debug import DebugImageWriter
from app.services.card_detection.orientation import rotate_image
from app.services.card_detection.roi import RoiCrop, extract_roi_crops
from app.services.card_detection.service import CardGeometryService
from app.services.card_detection.types import CardNotDetectedError, NormalizedCard
from app.services.ocr.base import DebugSink, OCRProvider, OCRResult
from app.services.recognition.image_io import decode_image_bytes, downscale_to_max_side
from app.services.recognition.readings import (
    OcrReadings,
    PrefixedDebugSink,
    field_from_ocr,
    has_plausible_set_code,
    reading_is_set_code,
)
from app.services.resolver.name_index import NameIndex
from app.services.resolver.resolver import Resolver
from app.services.visual.base import VisualRecognizer

__all__ = ["RecognitionService", "OcrReadings", "has_plausible_set_code", "reading_is_set_code"]

logger = logging.getLogger(__name__)

EARLY_STOP_CONFIDENCE = 0.90
"""Stop trying further ROIs once a reading reaches this confidence."""
RETRY_ROTATION_DEGREES = 180
DEBUG_KEY_NAME_ROI = "04_name_roi_{index}"
DEBUG_KEY_SET_CODE_ROI = "05_set_code_roi_{index}"
RETRY_DEBUG_PREFIX = "r180_"


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 2)


class RecognitionService:
    """Orchestrates geometry, OCR, resolution and the optional visual fallback."""

    def __init__(
        self,
        settings: Settings,
        *,
        geometry: CardGeometryService,
        ocr: OCRProvider,
        resolver: Resolver,
        visual: VisualRecognizer,
        layout: CardLayoutTemplate | None = None,
    ) -> None:
        self._settings = settings
        self._geometry = geometry
        self._ocr = ocr
        self._resolver = resolver
        self._visual = visual
        self._layout = layout or get_layout(settings.card_layout)

    # ---------------------------------------------------------------- accessors
    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def layout(self) -> CardLayoutTemplate:
        return self._layout

    @property
    def ocr_provider(self) -> OCRProvider:
        return self._ocr

    @property
    def visual_recognizer(self) -> VisualRecognizer:
        return self._visual

    @property
    def resolver(self) -> Resolver:
        return self._resolver

    @property
    def name_index(self) -> NameIndex:
        return self._resolver.name_index

    def refresh_name_index(self, repository: CardRepository) -> int:
        """Rebuild the fuzzy name index from ``repository`` (after a data sync); returns its size."""
        self._resolver.name_index.refresh(repository)
        return len(self._resolver.name_index)

    def warmup(self) -> None:
        """Load the OCR model eagerly (optional; otherwise it loads on the first request)."""
        self._ocr.warmup()

    # ------------------------------------------------------------------ public
    def recognize_bytes(self, data: bytes, *, debug: DebugImageWriter | None = None) -> RecognitionResult:
        """Decode ``data`` (JPEG/PNG/WEBP, EXIF-aware) and run :meth:`recognize`.

        Raises :class:`~app.services.recognition.image_io.ImageDecodeError` for undecodable bytes.
        """
        image = downscale_to_max_side(decode_image_bytes(data), self._settings.max_image_side)
        return self.recognize(image, debug=debug)

    def recognize(self, image: np.ndarray, *, debug: DebugImageWriter | None = None) -> RecognitionResult:
        """Recognize the single card in a BGR photo.  Never raises for a bad photo."""
        timing: dict[str, float] = {}
        started = time.perf_counter()

        try:
            card = self._geometry.process(image, debug=debug)
        except (CardNotDetectedError, ValueError) as exc:
            timing["detection"] = _ms(started)
            logger.info("Card not detected: %s", exc)
            return self._not_detected(str(exc), debug, timing, started)
        except Exception as exc:  # a crash in OpenCV must not surface as a 500 for one photo
            timing["detection"] = _ms(started)
            logger.exception("Card detection failed unexpectedly")
            return self._not_detected(f"card detection failed unexpectedly ({type(exc).__name__}: {exc})", debug, timing, started)
        timing["detection"] = _ms(started)

        ocr_started = time.perf_counter()
        try:
            readings, card = self._read_with_retry(card, debug)
        except Exception as exc:
            timing["ocr"] = _ms(ocr_started)
            logger.exception("OCR stage failed unexpectedly")
            return self._failed(card, None, f"OCR failed unexpectedly ({type(exc).__name__}: {exc})", debug, timing, started)
        timing["ocr"] = _ms(ocr_started)

        resolve_started = time.perf_counter()
        try:
            resolution = self._resolve(readings, card)
        except Exception as exc:
            timing["resolve"] = _ms(resolve_started)
            logger.exception("Resolver failed unexpectedly")
            return self._failed(card, readings, f"resolver failed unexpectedly ({type(exc).__name__}: {exc})", debug, timing, started)
        timing["resolve"] = _ms(resolve_started)
        timing["total"] = _ms(started)

        result = self._to_result(resolution, card, readings, debug, timing)
        logger.debug(
            "recognized status=%s confidence=%.3f card=%s printing=%s timing=%s",
            result.status.value, result.confidence,
            result.card.name if result.card else None,
            result.printing.set_code if result.printing else None,
            timing,
        )
        return result

    # --------------------------------------------------------------------- ocr
    def _read_with_retry(self, card: NormalizedCard, debug: DebugImageWriter | None) -> tuple[OcrReadings, NormalizedCard]:
        readings = self._read_card(card, debug)
        if not readings.is_weak(self._settings.ocr_min_confidence):
            return readings, card
        logger.info(
            "Weak OCR readings (name %.2f, set code %r); retrying with the card rotated %d degrees",
            readings.name_confidence, readings.set_code.text if readings.set_code else None, RETRY_ROTATION_DEGREES,
        )
        rotated = NormalizedCard(
            image=rotate_image(card.image, RETRY_ROTATION_DEGREES),
            hires=rotate_image(card.hires, RETRY_ROTATION_DEGREES),
            scale=card.scale,
            rotation_applied=(card.rotation_applied + RETRY_ROTATION_DEGREES) % 360,
            detection=card.detection,
        )
        retry_sink = PrefixedDebugSink(debug, RETRY_DEBUG_PREFIX) if debug is not None else None
        retry = self._read_card(rotated, retry_sink)
        retry.extra_rotation = RETRY_ROTATION_DEGREES
        if retry.strength() > readings.strength():
            logger.info("Rotated retry read better (%.2f > %.2f); keeping it", retry.strength(), readings.strength())
            return retry, rotated
        return readings, card

    def _read_card(self, card: NormalizedCard, sink: DebugSink | None) -> OcrReadings:
        crops = extract_roi_crops(card.hires, self._layout, card.scale)
        name, name_roi = self._read_name(crops["name"], sink)
        set_code, set_code_roi = self._read_set_code(crops["set_code"], sink)
        return OcrReadings(name=name, set_code=set_code, name_roi=name_roi, set_code_roi=set_code_roi)

    @staticmethod
    def _save(sink: DebugSink | None, key: str, image: np.ndarray) -> None:
        if sink is None or image.size == 0:
            return
        try:
            sink.save(key, image)
        except Exception:  # debug output must never break recognition
            logger.warning("Could not save debug image %s", key, exc_info=True)

    def _read_name(self, crops: list[RoiCrop], sink: DebugSink | None) -> tuple[OCRResult | None, str | None]:
        """Most confident non-empty name reading over the layout's name ROIs."""
        best: OCRResult | None = None
        best_roi: str | None = None
        for index, crop in enumerate(crops):
            self._save(sink, DEBUG_KEY_NAME_ROI.format(index=index), crop.image)
            result = self._ocr.recognize_name(crop.image, debug=sink)
            logger.debug("name roi %s -> %r (%.3f)", crop.roi.key, result.text, result.confidence)
            if result.is_empty:
                continue
            if best is None or result.confidence > best.confidence:
                best, best_roi = result, crop.roi.key
            if result.confidence >= EARLY_STOP_CONFIDENCE:
                break
        if best is None and crops:
            best = OCRResult.empty()
        return best, best_roi

    def _read_set_code(self, crops: list[RoiCrop], sink: DebugSink | None) -> tuple[OCRResult | None, str | None]:
        """Best set-code reading: set-code-shaped readings win over more confident shapeless ones."""
        best: OCRResult | None = None
        best_key: tuple[bool, bool, float] = (False, False, -1.0)
        best_roi: str | None = None
        for index, crop in enumerate(crops):
            self._save(sink, DEBUG_KEY_SET_CODE_ROI.format(index=index), crop.image)
            result = self._ocr.recognize_set_code(crop.image, debug=sink)
            logger.debug("set code roi %s -> %r (%.3f)", crop.roi.key, result.text, result.confidence)
            if result.is_empty:
                continue
            shaped = reading_is_set_code(result)
            key = (shaped, has_plausible_set_code(result), result.confidence)
            if key > best_key:
                best, best_key, best_roi = result, key, crop.roi.key
            if shaped and result.confidence >= EARLY_STOP_CONFIDENCE:
                break
        if best is None and crops:
            best = OCRResult.empty()
        return best, best_roi

    # ----------------------------------------------------------------- resolve
    def _resolve(self, readings: OcrReadings, card: NormalizedCard) -> ResolutionResult:
        resolution = self._resolver.resolve(readings.name, readings.set_code)
        if resolution.confidence < self._settings.visual_trigger_below_confidence and self._visual.available:
            candidates = self._visual.recognize(card.image)
            logger.info("Visual recognizer %s proposed %d candidate(s)", self._visual.name, len(candidates))
            if candidates:
                resolution = self._resolver.resolve(readings.name, readings.set_code, visual_candidates=candidates)
                resolution.notes.append(f"visual recognizer '{self._visual.name}' contributed {len(candidates)} candidate(s)")
        return resolution

    # ----------------------------------------------------------------- results
    def _detection_info(self, card: NormalizedCard | None) -> DetectionInfo:
        if card is None or card.detection is None:
            return DetectionInfo(
                detected=card is not None,
                rotation_applied=card.rotation_applied if card is not None else 0,
                layout=self._layout.key,
            )
        return DetectionInfo(
            detected=True,
            method=card.detection.method,
            corners=card.detection.corners_as_list(),
            rotation_applied=card.rotation_applied,
            layout=self._layout.key,
        )

    @staticmethod
    def _debug_paths(debug: DebugImageWriter | None) -> dict[str, str] | None:
        return debug.paths() if debug is not None else None

    def _not_detected(self, message: str, debug: DebugImageWriter | None, timing: dict[str, float], started: float) -> RecognitionResult:
        timing["total"] = _ms(started)
        return RecognitionResult(
            status=RecognitionStatus.CARD_NOT_DETECTED,
            notes=[message],
            detection=DetectionInfo(detected=False, layout=self._layout.key),
            debug=self._debug_paths(debug),
            timing_ms=timing,
        )

    def _failed(
        self,
        card: NormalizedCard,
        readings: OcrReadings | None,
        message: str,
        debug: DebugImageWriter | None,
        timing: dict[str, float],
        started: float,
    ) -> RecognitionResult:
        timing["total"] = _ms(started)
        ocr = RecognitionOcr()
        if readings is not None:
            ocr = RecognitionOcr(
                name=field_from_ocr(readings.name, kind="name"),
                set_code=field_from_ocr(readings.set_code, kind="set_code"),
            )
        return RecognitionResult(
            status=RecognitionStatus.OCR_FAILED,
            ocr=ocr,
            notes=[message],
            detection=self._detection_info(card),
            debug=self._debug_paths(debug),
            timing_ms=timing,
        )

    def _to_result(
        self,
        resolution: ResolutionResult,
        card: NormalizedCard,
        readings: OcrReadings,
        debug: DebugImageWriter | None,
        timing: dict[str, float],
    ) -> RecognitionResult:
        notes = list(resolution.notes)
        if readings.extra_rotation:
            notes.append(
                f"initial OCR readings were weak; the card was rotated {readings.extra_rotation} degrees and read again"
            )
        return RecognitionResult(
            status=resolution.status,
            card=resolution.card,
            printing=resolution.printing,
            ocr=RecognitionOcr(
                name=resolution.name or field_from_ocr(readings.name, kind="name"),
                set_code=resolution.set_code or field_from_ocr(readings.set_code, kind="set_code"),
            ),
            confidence=resolution.confidence,
            candidates=resolution.candidates,
            notes=notes,
            detection=self._detection_info(card),
            debug=self._debug_paths(debug),
            timing_ms=timing,
        )
