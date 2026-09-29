"""PaddleOCR (PP-OCRv5/v6 text recognition) provider.

Only the *recognition* model is used: the geometry stage already hands us a
tightly cropped text strip, so no text detection is needed.  Every
preprocessing variant of a crop is pushed through ``TextRecognition.predict``
in one batch and the most plausible reading is selected.

The engine is created lazily on first use (loading takes seconds and pulls in
Paddle), guarded by a lock so a multi-threaded API server never builds it
twice.  Inference is serialized behind the same lock because Paddle predictors
are not documented as thread-safe.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Protocol

import numpy as np

from app.services.ocr.base import DebugSink, OCRProvider, OCRResult
from app.services.ocr.preprocessing import ensure_bgr
from app.services.ocr.selection import Reading, ReadingKind, select_best_reading
from app.services.ocr.variants import NAME_VARIANTS, SET_CODE_VARIANTS, Variant

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "PP-OCRv6_medium_rec"


class TextRecognitionEngine(Protocol):
    """The subset of ``paddleocr.TextRecognition`` we rely on."""

    def predict(self, input: Any, batch_size: int = 1) -> Iterable[Any]: ...


def _quiet_paddle_environment() -> None:
    """Silence Paddle/PaddleX console chatter.  Must run BEFORE importing paddleocr."""
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("FLAGS_logtostderr", "0")
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "1")
    for noisy in ("paddlex", "paddleocr", "paddle"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def load_text_recognition(model_name: str, device: str) -> TextRecognitionEngine:
    """Import paddleocr and build a ``TextRecognition`` model (slow; called lazily)."""
    _quiet_paddle_environment()
    from paddleocr import TextRecognition  # noqa: PLC0415 - deliberate lazy import

    try:  # PaddleX re-configures its logger at import time; lower it again afterwards.
        from paddlex.utils import logging as paddlex_logging  # noqa: PLC0415

        paddlex_logging.setup_logging("WARNING")
    except Exception:  # pragma: no cover - cosmetic only
        logger.debug("Could not adjust paddlex logging", exc_info=True)
    logging.getLogger("paddlex").setLevel(logging.WARNING)
    logger.info("Loading PaddleOCR text recognition model %s on %s", model_name, device)
    return TextRecognition(model_name=model_name, device=device)


def _result_payload(result: Any) -> dict[str, Any]:
    """Extract the ``{"rec_text", "rec_score", ...}`` dict from a predict() item."""
    payload: Any = result
    json_attr = getattr(result, "json", None)
    if json_attr is not None:
        payload = json_attr() if callable(json_attr) else json_attr
    if isinstance(payload, dict) and "res" in payload and isinstance(payload["res"], dict):
        payload = payload["res"]
    if not isinstance(payload, dict):
        return {"rec_text": "", "rec_score": 0.0}
    # Drop bulky array values (input_img) so raw results stay JSON friendly.
    return {k: v for k, v in payload.items() if not isinstance(v, np.ndarray)}


def _is_empty_image(image: np.ndarray | None) -> bool:
    return image is None or image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0


class PaddleOCRProvider(OCRProvider):
    """OCR provider backed by PaddleOCR's recognition-only model.

    Parameters
    ----------
    model_name, device
        Passed to ``paddleocr.TextRecognition``.
    name_variants, set_code_variants
        Override the preprocessing variants (defaults from :mod:`variants`).
    engine_factory
        Test hook: a zero-argument callable returning something with a
        ``predict(input=..., batch_size=...)`` method.
    """

    name = "paddle"

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cpu",
        name_variants: Sequence[Variant] | None = None,
        set_code_variants: Sequence[Variant] | None = None,
        engine_factory: Callable[[], TextRecognitionEngine] | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.name_variants: list[Variant] = list(name_variants if name_variants is not None else NAME_VARIANTS)
        self.set_code_variants: list[Variant] = list(
            set_code_variants if set_code_variants is not None else SET_CODE_VARIANTS
        )
        self._engine_factory = engine_factory or (lambda: load_text_recognition(self.model_name, self.device))
        self._engine: TextRecognitionEngine | None = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------ engine
    @property
    def is_loaded(self) -> bool:
        """True once the recognition model has been created."""
        return self._engine is not None

    def warmup(self) -> None:
        """Load the model now instead of on the first recognition call."""
        self._get_engine()

    def _get_engine(self) -> TextRecognitionEngine:
        if self._engine is None:
            with self._lock:
                if self._engine is None:
                    self._engine = self._engine_factory()
        return self._engine

    # ------------------------------------------------------------ public API
    def recognize_name(self, image: np.ndarray, *, debug: DebugSink | None = None) -> OCRResult:
        return self._recognize("name", image, self.name_variants, debug)

    def recognize_set_code(self, image: np.ndarray, *, debug: DebugSink | None = None) -> OCRResult:
        return self._recognize("set_code", image, self.set_code_variants, debug)

    def recognize_text(self, image: np.ndarray) -> OCRResult:
        """Recognize the raw crop only (no variants, no plausibility ranking)."""
        if _is_empty_image(image):
            return OCRResult.empty()
        try:
            readings = self._predict_batch([("raw", ensure_bgr(image))])
        except Exception:
            logger.exception("PaddleOCR recognize_text failed")
            return OCRResult.empty()
        if not readings:
            return OCRResult.empty()
        variant, text, confidence, raw = readings[0]
        if not text.strip():
            return OCRResult.empty(raw_result=raw)
        return OCRResult(text=text, confidence=confidence, raw_result=raw, variant=variant)

    # ------------------------------------------------------------ internals
    def _build_variant_images(
        self, kind: ReadingKind, image: np.ndarray, variants: Sequence[Variant], debug: DebugSink | None
    ) -> list[tuple[str, np.ndarray]]:
        prepared: list[tuple[str, np.ndarray]] = []
        for variant_name, fn in variants:
            try:
                out = ensure_bgr(fn(image))
            except Exception:
                logger.warning("OCR variant %s/%s failed; skipping", kind, variant_name, exc_info=True)
                continue
            if _is_empty_image(out):
                logger.debug("OCR variant %s/%s produced an empty image; skipping", kind, variant_name)
                continue
            if not out.flags.c_contiguous:
                out = np.ascontiguousarray(out)
            if debug is not None:
                try:
                    debug.save(f"ocr_{kind}_{variant_name}", out)
                except Exception:  # debug output must never break recognition
                    logger.warning("Could not save debug image for %s/%s", kind, variant_name, exc_info=True)
            prepared.append((variant_name, out))
        return prepared

    def _recognize(
        self, kind: ReadingKind, image: np.ndarray, variants: Sequence[Variant], debug: DebugSink | None
    ) -> OCRResult:
        if _is_empty_image(image):
            logger.debug("Empty %s crop; returning empty OCR result", kind)
            return OCRResult.empty()
        prepared = self._build_variant_images(kind, image, variants, debug)
        if not prepared:
            return OCRResult.empty()
        try:
            readings = self._predict_batch(prepared)
        except Exception:
            logger.exception("PaddleOCR %s recognition failed", kind)
            return OCRResult.empty()
        result = select_best_reading(readings, kind)
        logger.debug(
            "OCR %s -> %r (%.3f, variant=%s); %d alternative(s)",
            kind, result.text, result.confidence, result.variant, len(result.alternatives),
        )
        return result

    def _predict_batch(self, items: Sequence[tuple[str, np.ndarray]]) -> list[Reading]:
        """Run every prepared variant through the engine in one call."""
        engine = self._get_engine()
        images = [img for _, img in items]
        with self._lock:
            results = list(engine.predict(input=images, batch_size=max(1, len(images))))
        if len(results) != len(items):
            logger.warning("PaddleOCR returned %d results for %d inputs", len(results), len(items))
        readings: list[Reading] = []
        for (variant_name, _), result in zip(items, results):
            payload = _result_payload(result)
            text = str(payload.get("rec_text") or "")
            score = float(payload.get("rec_score") or 0.0)
            readings.append((variant_name, text, score, payload))
        return readings
