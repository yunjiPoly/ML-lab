"""FastAPI application: ``uvicorn app.main:app --port 8000``.

:func:`create_app` builds an application for the given settings (tests pass
their own); the lifespan opens the database, builds the recognition service
once and stores both on ``app.state``.  The OCR model is loaded lazily on the
first request unless ``OCR_WARMUP_ON_STARTUP=true``, so tests and the CLI
start instantly.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.errors import register_exception_handlers
from app.api.router import api_router
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.db.session import Database
from app.services.recognition.factory import SessionScopedRepository, build_recognition_service
from app.services.recognition.pipeline import RecognitionService

logger = logging.getLogger(__name__)

DESCRIPTION = (
    "Photo of a physical Yu-Gi-Oh! card in, exact card and printing out: OpenCV card detection, "
    "perspective correction, PaddleOCR on the name and set-code regions, then a database resolver."
)


def _warm_ocr(service: RecognitionService) -> None:
    try:
        service.warmup()
        logger.info("OCR model warm-up finished")
    except Exception:  # never take the server down over a warm-up failure
        logger.exception("OCR model warm-up failed; the model will be retried on first use")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    if not logging.getLogger().handlers:  # leave pytest / uvicorn logging configurations alone
        configure_logging("DEBUG" if settings.debug else settings.log_level, settings.log_json)
    database = Database(settings.resolved_database_url, echo=settings.sql_echo)
    database.create_all()
    service = build_recognition_service(settings, database)
    app.state.database = database
    app.state.repository = SessionScopedRepository(database)
    app.state.service = service
    if settings.ocr_warmup_on_startup:
        threading.Thread(target=_warm_ocr, args=(service,), name="ocr-warmup", daemon=True).start()
    logger.info(
        "%s v%s started: database=%s ocr=%s visual=%s cards=%d",
        settings.app_name, __version__, database.url, service.ocr_provider.name,
        service.visual_recognizer.name, len(service.name_index),
    )
    try:
        yield
    finally:
        database.dispose()
        logger.info("Application shut down")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory (tests call it with isolated settings)."""
    settings = settings or get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
        debug=False,  # keep the JSON 500 handler even in DEBUG mode
    )
    app.state.settings = settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_origin_regex=settings.cors_origin_regex or None,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_exception_handlers(app)
    app.include_router(api_router)
    return app


app = create_app()
