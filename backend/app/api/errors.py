"""Exception handlers: consistent ``{"detail": ...}`` JSON bodies for every error."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.services.recognition.image_io import ImageDecodeError, UploadValidationError

logger = logging.getLogger(__name__)


async def upload_validation_handler(_request: Request, exc: UploadValidationError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


async def image_decode_handler(_request: Request, exc: ImageDecodeError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": f"could not decode the uploaded image: {exc}"})


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(status_code=500, content={"detail": "internal error"})


def register_exception_handlers(app: FastAPI) -> None:
    """Attach the handlers to ``app`` (called by :func:`app.main.create_app`)."""
    app.add_exception_handler(UploadValidationError, upload_validation_handler)  # type: ignore[arg-type]
    app.add_exception_handler(ImageDecodeError, image_decode_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unhandled_exception_handler)
