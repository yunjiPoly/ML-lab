"""FastAPI dependencies.

Everything long-lived (settings, database, recognition service) is created by
the application lifespan and stored on ``app.state``; the dependencies here
only fetch it.  :func:`get_repository` opens one SQLAlchemy session per request
for the card lookup endpoints.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Request

from app.core.config import Settings
from app.db.repository import SQLAlchemyCardRepository
from app.db.session import Database
from app.services.recognition.pipeline import RecognitionService


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_database(request: Request) -> Database:
    database = getattr(request.app.state, "database", None)
    if database is None:
        raise RuntimeError("database not initialised: the application lifespan has not run")
    return database


def get_service(request: Request) -> RecognitionService:
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise RuntimeError("recognition service not initialised: the application lifespan has not run")
    return service


def get_repository(database: Annotated[Database, Depends(get_database)]) -> Iterator[SQLAlchemyCardRepository]:
    """Per-request repository bound to a session that is closed when the request ends."""
    with database.session() as session:
        yield SQLAlchemyCardRepository(session)


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
DatabaseDep = Annotated[Database, Depends(get_database)]
ServiceDep = Annotated[RecognitionService, Depends(get_service)]
RepositoryDep = Annotated[SQLAlchemyCardRepository, Depends(get_repository)]
