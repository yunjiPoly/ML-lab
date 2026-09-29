"""All API routes mounted under ``/api``."""

from __future__ import annotations

from fastapi import APIRouter

from app.api import routes_cards, routes_health, routes_recognize

api_router = APIRouter(prefix="/api")
api_router.include_router(routes_health.router)
api_router.include_router(routes_recognize.router)
api_router.include_router(routes_cards.router)

__all__ = ["api_router"]
