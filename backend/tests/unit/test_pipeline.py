"""Unit tests for ``RecognitionService`` with a stub geometry service and fake OCR providers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.core.config import Settings
from app.schemas.recognition import RecognitionStatus
from app.services.card_detection.debug import DebugImageWriter
from app.services.card_detection.service import CardGeometryService
from app.services.card_detection.types import CardDetection, CardNotDetectedError, NormalizedCard
from app.services.ocr.base import DebugSink, OCRAlternative, OCRProvider, OCRResult
from app.services.ocr.fake_provider import FakeOCRProvider
from app.services.recognition.image_io import ImageDecodeError
from app.services.recognition.pipeline import OcrReadings, RecognitionService, has_plausible_set_code
from app.services.resolver.resolver import Resolver
from app.services.visual.base import NoOpVisualRecognizer, VisualCandidate, VisualRecognizer
from tests.fixtures.fake_repository import FakeCardRepository

ARMADES_ID = 88033975
ARMADES = "Armades, Keeper of Boundaries"


# ------------------------------------------------------------------- stubs
class StubGeometryService(CardGeometryService):
    """Returns a synthetic normalized card (or raises) without running OpenCV detection.

    The high-resolution card has a bright top band and a dark remainder so a
    test can tell whether a crop came from the original or the 180-degree
    rotated card.
    """

    def __init__(self, settings: Settings, *, fail: bool = False, rotation: int = 0) -> None:
        super().__init__(settings)
        self.fail = fail
        self.rotation = rotation
        self.calls = 0

    def process(self, image: np.ndarray, *, debug: DebugImageWriter | None = None) -> NormalizedCard:
        self.calls += 1
        if self.fail:
            raise CardNotDetectedError("No card-shaped quadrilateral was found in the image (stub)")
        s = self._settings
        canonical = np.full((s.card_height, s.card_width, 3), 240, dtype=np.uint8)
        hires = np.zeros((s.card_height * s.ocr_scale, s.card_width * s.ocr_scale, 3), dtype=np.uint8)
        hires[: hires.shape[0] // 5] = 255  # bright top band
        if debug is not None:
            debug.save("01_detected_contour", image)
        corners = np.array([[10, 10], [400, 12], [398, 600], [8, 598]], dtype=np.float32)
        return NormalizedCard(
            image=canonical,
            hires=hires,
            scale=s.ocr_scale,
            rotation_applied=self.rotation,
            detection=CardDetection(corners=corners, method="stub", score=0.9),
        )


class ScriptedOCR(OCRProvider):
    """Returns scripted readings in call order (the last entry repeats) and records crop brightness."""

    name = "scripted"

    def __init__(self, names: Sequence[OCRResult], codes: Sequence[OCRResult]) -> None:
        self._names = list(names)
        self._codes = list(codes)
        self.name_means: list[float] = []
        self.code_means: list[float] = []

    @staticmethod
    def _next(queue: list[OCRResult]) -> OCRResult:
        return queue.pop(0) if len(queue) > 1 else queue[0]

    def recognize_name(self, image: np.ndarray, *, debug: DebugSink | None = None) -> OCRResult:
        self.name_means.append(float(image.mean()))
        return self._next(self._names)

    def recognize_set_code(self, image: np.ndarray, *, debug: DebugSink | None = None) -> OCRResult:
        self.code_means.append(float(image.mean()))
        return self._next(self._codes)

    def recognize_text(self, image: np.ndarray) -> OCRResult:
        return OCRResult.empty()


class ExplodingOCR(OCRProvider):
    name = "exploding"

    def recognize_name(self, image: np.ndarray, *, debug: DebugSink | None = None) -> OCRResult:
        raise RuntimeError("engine crashed")

    def recognize_set_code(self, image: np.ndarray, *, debug: DebugSink | None = None) -> OCRResult:
        raise RuntimeError("engine crashed")

    def recognize_text(self, image: np.ndarray) -> OCRResult:
        raise RuntimeError("engine crashed")


class StubVisual(VisualRecognizer):
    name = "stub-visual"

    def __init__(self, candidates: Sequence[VisualCandidate]) -> None:
        self.candidates = list(candidates)
        self.calls = 0

    def recognize(self, image: np.ndarray) -> list[VisualCandidate]:
        self.calls += 1
        return list(self.candidates)


# ---------------------------------------------------------------- fixtures
@pytest.fixture()
def repo(ygoprodeck_sample: dict) -> FakeCardRepository:
    return FakeCardRepository.from_payload(ygoprodeck_sample)


def make_service(
    settings: Settings,
    repo: FakeCardRepository,
    ocr: OCRProvider,
    *,
    geometry: CardGeometryService | None = None,
    visual: VisualRecognizer | None = None,
) -> RecognitionService:
    return RecognitionService(
        settings,
        geometry=geometry or StubGeometryService(settings),
        ocr=ocr,
        resolver=Resolver(repo, settings),
        visual=visual or NoOpVisualRecognizer(),
    )


def photo() -> np.ndarray:
    return np.full((300, 200, 3), 128, dtype=np.uint8)


def armades_ocr() -> FakeOCRProvider:
    return FakeOCRProvider(
        name_text="ARMADES KEEPER OF BOUNDARIES",
        name_confidence=0.7,
        set_code_text="J0TL-ENO45",
        set_code_confidence=0.95,
    )


# ------------------------------------------------------------------- tests
def test_matched_armades_from_canonical_readings(test_settings: Settings, repo: FakeCardRepository) -> None:
    ocr = armades_ocr()
    result = make_service(test_settings, repo, ocr).recognize(photo())

    assert result.status is RecognitionStatus.MATCHED
    assert result.card is not None and result.card.id == ARMADES_ID and result.card.name == ARMADES
    assert result.printing is not None
    assert (result.printing.set_code, result.printing.set_name, result.printing.rarity) == (
        "JOTL-EN045", "Judgment of the Light", "Secret Rare",
    )
    assert result.confidence >= 0.8
    assert result.ocr.name is not None and result.ocr.name.raw == "ARMADES KEEPER OF BOUNDARIES"
    assert result.ocr.set_code is not None and result.ocr.set_code.raw == "J0TL-ENO45"
    assert result.ocr.set_code.normalized == "JOTL-EN045"
    assert result.detection is not None and result.detection.detected
    assert result.detection.method == "stub" and result.detection.rotation_applied == 0
    assert result.detection.layout == "standard" and len(result.detection.corners or []) == 4
    assert set(result.timing_ms) == {"detection", "ocr", "resolve", "total"}
    assert result.timing_ms["total"] >= 0.0
    assert result.debug is None
    # one name ROI, then the first set-code ROI already reads a plausible code at >= 0.9 -> early stop
    assert ocr.calls == ["name", "set_code"]


def test_card_not_detected_when_geometry_raises(test_settings: Settings, repo: FakeCardRepository) -> None:
    ocr = armades_ocr()
    geometry = StubGeometryService(test_settings, fail=True)
    result = make_service(test_settings, repo, ocr, geometry=geometry).recognize(photo())

    assert result.status is RecognitionStatus.CARD_NOT_DETECTED
    assert result.card is None and result.printing is None
    assert result.detection is not None and not result.detection.detected
    assert any("No card-shaped quadrilateral" in note for note in result.notes)
    assert ocr.calls == []  # OCR never ran
    assert {"detection", "total"} <= set(result.timing_ms)


def test_ocr_failed_when_fake_returns_empty(test_settings: Settings, repo: FakeCardRepository) -> None:
    ocr = FakeOCRProvider()
    result = make_service(test_settings, repo, ocr).recognize(photo())

    assert result.status is RecognitionStatus.OCR_FAILED
    assert result.card is None and result.printing is None
    assert result.ocr.name is not None and result.ocr.name.raw == ""
    assert any("no usable name or set code" in note for note in result.notes)
    # both readings were weak -> the card was rotated and read again: 1 name + 2 set-code ROIs per pass
    assert ocr.calls.count("name") == 2 and ocr.calls.count("set_code") == 4
    assert result.detection is not None and result.detection.rotation_applied == 0  # retry was not better


def test_weak_readings_trigger_rotated_retry(test_settings: Settings, repo: FakeCardRepository) -> None:
    ocr = ScriptedOCR(
        names=[OCRResult("~~", 0.10), OCRResult("ARMADES, KEEPER OF BOUNDARIES", 0.95, variant="tight")],
        codes=[OCRResult.empty(), OCRResult.empty(), OCRResult("JOTL-EN045", 0.99, variant="raw")],
    )
    result = make_service(test_settings, repo, ocr).recognize(photo())

    assert result.status is RecognitionStatus.MATCHED
    assert result.card is not None and result.card.id == ARMADES_ID
    assert result.printing is not None and result.printing.set_code == "JOTL-EN045"
    assert result.detection is not None and result.detection.rotation_applied == 180
    assert any("rotated 180 degrees" in note for note in result.notes)
    # the name crop came from the bright top band first, then from the dark band of the rotated card
    assert len(ocr.name_means) == 2
    assert ocr.name_means[0] > 200 and ocr.name_means[1] < 50


def test_set_code_shaped_reading_beats_more_confident_noise(test_settings: Settings, repo: FakeCardRepository) -> None:
    ocr = ScriptedOCR(
        names=[OCRResult("ARMADES, KEEPER OF BOUNDARIES", 0.95)],
        codes=[OCRResult("1st Edition", 0.97), OCRResult("JOTL-EN045", 0.80)],
    )
    result = make_service(test_settings, repo, ocr).recognize(photo())

    assert result.ocr.set_code is not None and result.ocr.set_code.raw == "JOTL-EN045"
    assert result.status is RecognitionStatus.MATCHED
    assert result.printing is not None and result.printing.set_code == "JOTL-EN045"
    assert len(ocr.code_means) == 2  # both set-code ROIs were tried, the shaped one won


def test_debug_images_are_saved(test_settings: Settings, repo: FakeCardRepository, tmp_path: Path) -> None:
    ocr = FakeOCRProvider(name_text=ARMADES, name_confidence=0.95, set_code_text="JOTL-EN045", set_code_confidence=0.99)
    writer = DebugImageWriter(tmp_path / "dbg")
    result = make_service(test_settings, repo, ocr).recognize(photo(), debug=writer)

    assert result.status is RecognitionStatus.MATCHED
    assert result.debug is not None
    assert {"01_detected_contour", "04_name_roi_0", "05_set_code_roi_0", "ocr_name_raw", "ocr_set_code_raw"} <= set(result.debug)
    assert "05_set_code_roi_1" not in result.debug  # early stop after the first plausible set code
    assert all(Path(path).is_file() for path in result.debug.values())


def test_visual_fallback_when_text_matches_nothing(test_settings: Settings, repo: FakeCardRepository) -> None:
    ocr = FakeOCRProvider(name_text="ZZZZ QQQQ WWWW", name_confidence=0.9, set_code_text="XXXX-XX999", set_code_confidence=0.9)
    visual = StubVisual([VisualCandidate(card_id=ARMADES_ID, score=0.8)])
    result = make_service(test_settings, repo, ocr, visual=visual).recognize(photo())

    assert visual.calls == 1
    assert result.status is RecognitionStatus.LOW_CONFIDENCE
    assert result.candidates and result.candidates[0].card.id == ARMADES_ID
    assert any("visual" in note for note in result.notes)


def test_visual_recognizer_not_called_when_confident(test_settings: Settings, repo: FakeCardRepository) -> None:
    visual = StubVisual([VisualCandidate(card_id=ARMADES_ID, score=0.8)])
    result = make_service(test_settings, repo, armades_ocr(), visual=visual).recognize(photo())
    assert result.status is RecognitionStatus.MATCHED
    assert visual.calls == 0


def test_ocr_exception_becomes_ocr_failed(test_settings: Settings, repo: FakeCardRepository) -> None:
    result = make_service(test_settings, repo, ExplodingOCR()).recognize(photo())
    assert result.status is RecognitionStatus.OCR_FAILED
    assert any("engine crashed" in note for note in result.notes)
    assert result.detection is not None and result.detection.detected
    assert {"detection", "ocr", "total"} <= set(result.timing_ms)


def test_recognize_bytes_decodes_then_recognizes(test_settings: Settings, repo: FakeCardRepository) -> None:
    ocr = FakeOCRProvider(name_text="Kuriboh", name_confidence=0.95)
    service = make_service(test_settings, repo, ocr)
    ok, encoded = cv2.imencode(".png", photo())
    assert ok
    result = service.recognize_bytes(encoded.tobytes())
    assert result.card is not None and result.card.name == "Kuriboh"
    assert result.status in (RecognitionStatus.MATCHED, RecognitionStatus.LOW_CONFIDENCE)
    with pytest.raises(ImageDecodeError):
        service.recognize_bytes(b"not an image")


def test_service_exposes_components_and_index_refresh(test_settings: Settings, repo: FakeCardRepository) -> None:
    ocr = armades_ocr()
    service = make_service(test_settings, repo, ocr)
    assert service.ocr_provider is ocr
    assert service.visual_recognizer.name == "noop"
    assert service.layout.key == "standard"
    assert len(service.name_index) == 7
    repo.add_card(900100, "Brand New Card", printings=[("NEW-EN001", "New Set", "Common", "C")])
    assert service.refresh_name_index(repo) == 8


def test_plausibility_helpers() -> None:
    assert has_plausible_set_code(OCRResult("JOTL-EN045", 0.5))
    assert has_plausible_set_code(OCRResult("J0TL-ENO45", 0.5))
    assert has_plausible_set_code(OCRResult("garbage", 0.5, alternatives=[OCRAlternative("JOTL-EN045", 0.4)]))
    assert not has_plausible_set_code(OCRResult("1st Edition", 0.99))
    assert not has_plausible_set_code(OCRResult.empty())
    assert not has_plausible_set_code(None)

    weak = OcrReadings(name=OCRResult("x", 0.1), set_code=OCRResult("nope", 0.9))
    assert weak.is_weak(0.3)
    strong_name = OcrReadings(name=OCRResult("Kuriboh", 0.8), set_code=None)
    assert not strong_name.is_weak(0.3)
    shaped_code = OcrReadings(name=None, set_code=OCRResult("LOB-001", 0.2))
    assert not shaped_code.is_weak(0.3)
    assert shaped_code.strength() > strong_name.strength()
