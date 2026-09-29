"""PaddleOCRProvider orchestration tested with a stub engine (Paddle is never imported)."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.services.ocr.paddle_provider import PaddleOCRProvider, _result_payload
from app.services.ocr.variants import NAME_VARIANTS, SET_CODE_VARIANTS, canvas_size


class StubResult:
    """Mimics a paddleocr predict() item: ``.json == {"res": {...}}``."""

    def __init__(self, text: str, score: float) -> None:
        self.json = {"res": {"input_path": None, "page_index": None, "rec_text": text, "rec_score": score}}


class StubEngine:
    """Returns one reading per input; records every predict() call."""

    def __init__(self, readings: dict[int, tuple[str, float]] | None = None, *, fail: bool = False) -> None:
        self.calls: list[list[np.ndarray]] = []
        self.readings = readings or {}
        self.fail = fail
        self.counter = 0

    def predict(self, input: list[np.ndarray], batch_size: int = 1) -> list[StubResult]:
        if self.fail:
            raise RuntimeError("engine exploded")
        self.calls.append(list(input))
        out = []
        for img in input:
            assert img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.uint8, "engine expects BGR uint8"
            text, score = self.readings.get(self.counter, (f"reading{self.counter}", 0.5))
            self.counter += 1
            out.append(StubResult(text, score))
        return out


class RecordingSink:
    def __init__(self) -> None:
        self.saved: dict[str, np.ndarray] = {}

    def save(self, key: str, image: np.ndarray) -> Path:
        self.saved[key] = image
        return Path(key)


@pytest.fixture()
def crop() -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.integers(0, 256, size=(55, 345, 3), dtype=np.uint8)


def test_lazy_engine_is_created_once_even_across_threads() -> None:
    created: list[int] = []

    def factory() -> StubEngine:
        created.append(1)
        return StubEngine()

    provider = PaddleOCRProvider(engine_factory=factory)
    assert provider.is_loaded is False
    threads = [threading.Thread(target=provider.warmup) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert provider.is_loaded is True and len(created) == 1


def test_name_variants_share_one_canvas_and_one_predict_call(crop: np.ndarray) -> None:
    engine = StubEngine({1: ("ARMADES, KEEPER OF BOUNDARIES", 0.99), 0: ("ARMADESKEEPEOBOUNDR", 0.71)})
    provider = PaddleOCRProvider(engine_factory=lambda: engine)
    sink = RecordingSink()
    result = provider.recognize_name(crop, debug=sink)

    assert len(engine.calls) == 1, "all shipped name variants must go through one predict() call"
    assert len(engine.calls[0]) == len(NAME_VARIANTS)
    expected = canvas_size(crop, 2.0)
    assert {img.shape for img in engine.calls[0]} == {(expected[1], expected[0], 3)}
    assert result.text == "ARMADES, KEEPER OF BOUNDARIES" and result.variant == NAME_VARIANTS[1][0]
    assert {a.text for a in result.alternatives} <= {"ARMADESKEEPEOBOUNDR", "reading2", "reading3", "reading4"}
    assert set(sink.saved) == {f"ocr_name_{name}" for name, _ in NAME_VARIANTS}
    assert result.raw_result["rec_text"] == "ARMADES, KEEPER OF BOUNDARIES"


def test_set_code_variants_one_call_and_plausibility_selection() -> None:
    crop = np.random.default_rng(4).integers(0, 256, size=(30, 115, 3), dtype=np.uint8)
    engine = StubEngine({0: ("1ST EDITION", 0.999), 3: ("JOTL-EN045", 0.95)})
    provider = PaddleOCRProvider(engine_factory=lambda: engine)
    sink = RecordingSink()
    result = provider.recognize_set_code(crop, debug=sink)
    assert len(engine.calls) == 1 and len(engine.calls[0]) == len(SET_CODE_VARIANTS)
    assert result.text == "JOTL-EN045" and result.variant == SET_CODE_VARIANTS[3][0]
    assert set(sink.saved) == {f"ocr_set_code_{name}" for name, _ in SET_CODE_VARIANTS}


def test_mixed_shape_custom_variants_are_grouped_per_shape(crop: np.ndarray) -> None:
    engine = StubEngine()
    variants = [
        ("a", lambda im: im),
        ("b", lambda im: im[:, :100]),
        ("c", lambda im: im.copy()),
    ]
    provider = PaddleOCRProvider(engine_factory=lambda: engine, name_variants=variants)
    result = provider.recognize_name(crop)
    assert len(engine.calls) == 2
    assert sorted(len(c) for c in engine.calls) == [1, 2]
    assert {result.text, *(a.text for a in result.alternatives)} == {"reading0", "reading1", "reading2"}


def test_gray_variant_output_is_converted_to_bgr(crop: np.ndarray) -> None:
    engine = StubEngine()
    provider = PaddleOCRProvider(engine_factory=lambda: engine, set_code_variants=[("gray", lambda im: im[:, :, 0])])
    provider.recognize_set_code(crop)
    assert engine.calls[0][0].shape == (55, 345, 3)


def test_failing_variant_is_skipped_not_fatal(crop: np.ndarray) -> None:
    engine = StubEngine({0: ("Kuriboh", 0.9)})

    def boom(_: np.ndarray) -> np.ndarray:
        raise ValueError("bad variant")

    provider = PaddleOCRProvider(engine_factory=lambda: engine, name_variants=[("boom", boom), ("ok", lambda im: im)])
    result = provider.recognize_name(crop)
    assert result.text == "Kuriboh" and result.variant == "ok"


def test_empty_crops_and_engine_errors_yield_empty_results(crop: np.ndarray) -> None:
    provider = PaddleOCRProvider(engine_factory=lambda: StubEngine(fail=True))
    assert provider.recognize_name(np.zeros((0, 10, 3), np.uint8)).is_empty
    assert provider.recognize_set_code(np.zeros((10, 0, 3), np.uint8)).is_empty
    assert provider.recognize_text(np.zeros((0, 0, 3), np.uint8)).is_empty
    # engine raising -> logged, empty result, no exception
    assert provider.recognize_name(crop).is_empty
    assert provider.recognize_set_code(crop).is_empty
    assert provider.recognize_text(crop).is_empty


def test_recognize_text_runs_raw_crop_only(crop: np.ndarray) -> None:
    engine = StubEngine({0: ("Raw Text", 0.42)})
    provider = PaddleOCRProvider(engine_factory=lambda: engine)
    result = provider.recognize_text(crop)
    assert len(engine.calls) == 1 and len(engine.calls[0]) == 1
    assert engine.calls[0][0].shape == crop.shape
    assert (result.text, result.confidence, result.variant, result.alternatives) == ("Raw Text", 0.42, "raw", [])


def test_result_payload_variants() -> None:
    assert _result_payload(StubResult("x", 0.5))["rec_text"] == "x"
    assert _result_payload({"res": {"rec_text": "y", "rec_score": 0.1}})["rec_text"] == "y"
    assert _result_payload({"rec_text": "z", "rec_score": 0.2, "input_img": np.zeros((2, 2))}) == {
        "rec_text": "z",
        "rec_score": 0.2,
    }
    assert _result_payload("nonsense") == {"rec_text": "", "rec_score": 0.0}

    class CallableJson:
        def json(self) -> dict[str, Any]:
            return {"res": {"rec_text": "call", "rec_score": 0.3}}

    assert _result_payload(CallableJson())["rec_text"] == "call"
