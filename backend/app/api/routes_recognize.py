"""``POST /api/recognize``: multipart photo in, :class:`RecognitionResult` out.

The upload is read fully into memory (never written to disk), validated by
magic bytes, decoded and handed to the :class:`RecognitionService` in a worker
thread so the event loop stays responsive while OpenCV / PaddleOCR run.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Query, Request
from starlette.concurrency import run_in_threadpool

from app.api.deps import ServiceDep, SettingsDep
from app.api.upload import read_upload
from app.core.config import Settings
from app.schemas.recognition import RecognitionResult
from app.services.card_detection.debug import DebugImageWriter
from app.services.recognition.image_io import decode_image_bytes, downscale_to_max_side, validate_upload
from app.services.recognition.pipeline import RecognitionService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["recognition"])

UPLOAD_FIELD = "image"

_MULTIPART_SCHEMA = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": [UPLOAD_FIELD],
                    "properties": {
                        UPLOAD_FIELD: {"type": "string", "format": "binary", "description": "JPEG, PNG or WEBP photo"}
                    },
                }
            }
        },
    }
}


def build_debug_writer(settings: Settings, *, subdir: str = "api") -> DebugImageWriter:
    """Writer for one request: ``<debug_dir>/api/<timestamp>-<id>/``."""
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}"
    return DebugImageWriter(Path(settings.resolved_debug_dir) / subdir / stamp)


def run_recognition(
    service: RecognitionService, settings: Settings, data: bytes, debug: DebugImageWriter | None
) -> RecognitionResult:
    """Decode + downscale + recognize (synchronous; runs in a worker thread)."""
    image = downscale_to_max_side(decode_image_bytes(data), settings.max_image_side)
    return service.recognize(image, debug=debug)


@router.post(
    "/recognize",
    response_model=RecognitionResult,
    summary="Recognize the card in a photo",
    openapi_extra=_MULTIPART_SCHEMA,
    responses={
        400: {"description": "Malformed multipart body"},
        413: {"description": "Upload larger than MAX_UPLOAD_BYTES"},
        415: {"description": "Not a JPEG/PNG/WEBP image"},
        422: {"description": "Image field missing or the image could not be decoded"},
    },
)
async def recognize(
    request: Request,
    settings: SettingsDep,
    service: ServiceDep,
    debug: bool = Query(
        False,
        description="Also write intermediate images (only honoured when SAVE_DEBUG_IMAGES or DEBUG is enabled).",
    ),
) -> RecognitionResult:
    started = time.perf_counter()
    upload = await read_upload(request, field=UPLOAD_FIELD, max_bytes=settings.max_upload_bytes)
    mime = validate_upload(upload.content_type, upload.size, settings, data=upload.data)
    writer = build_debug_writer(settings) if debug and (settings.save_debug_images or settings.debug) else None

    result = await run_in_threadpool(run_recognition, service, settings, upload.data, writer)

    logger.info(
        "recognize status=%s confidence=%.3f card=%s printing=%s bytes=%d mime=%s request_ms=%.1f timing=%s",
        result.status.value,
        result.confidence,
        result.card.id if result.card else None,
        result.printing.set_code if result.printing else None,
        upload.size,
        mime,
        (time.perf_counter() - started) * 1000.0,
        result.timing_ms,
        extra={
            "extra_fields": {
                "status": result.status.value,
                "confidence": result.confidence,
                "card_id": result.card.id if result.card else None,
                "set_code": result.printing.set_code if result.printing else None,
                "timing_ms": result.timing_ms,
            }
        },
    )
    return result
