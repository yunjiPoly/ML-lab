"""``GET /api/health``: liveness plus a summary of the configured components."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app import __version__
from app.api.deps import RepositoryDep, ServiceDep

router = APIRouter(tags=["health"])


class DatabaseStats(BaseModel):
    cards: int
    printings: int


class HealthOut(BaseModel):
    status: str
    version: str
    ocr_provider: str
    ocr_model_loaded: bool | None = None
    visual_recognizer: str
    database: DatabaseStats


@router.get("/health", response_model=HealthOut, summary="Service health and configuration summary")
def health(service: ServiceDep, repository: RepositoryDep) -> HealthOut:
    ocr = service.ocr_provider
    loaded = getattr(ocr, "is_loaded", None)
    return HealthOut(
        status="ok",
        version=__version__,
        ocr_provider=ocr.name,
        ocr_model_loaded=bool(loaded) if loaded is not None else None,
        visual_recognizer=service.visual_recognizer.name,
        database=DatabaseStats(cards=repository.count_cards(), printings=repository.count_printings()),
    )
