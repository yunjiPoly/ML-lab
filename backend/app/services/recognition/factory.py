"""Wire the recognition pipeline from :class:`Settings`.

Used by the API lifespan and the CLI.  The expensive, thread-safe parts
(geometry service, OCR provider with its lazily loaded model, name index) are
built once per process; database access goes through a
:class:`SessionScopedRepository` so the long-lived service never holds a
SQLAlchemy session.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from app.core.config import Settings
from app.db.protocols import CardRepository
from app.db.repository import SQLAlchemyCardRepository
from app.db.session import Database
from app.models import Card, Printing
from app.services.card_detection.service import CardGeometryService
from app.services.ocr.base import OCRProvider
from app.services.ocr.factory import build_ocr_provider
from app.services.recognition.pipeline import RecognitionService
from app.services.resolver.name_index import NameIndex
from app.services.resolver.resolver import Resolver
from app.services.visual.base import VisualRecognizer, build_visual_recognizer

logger = logging.getLogger(__name__)

RepositorySource = Database | Callable[[], CardRepository]
"""Either a :class:`Database` (wrapped in a :class:`SessionScopedRepository`) or a factory."""


class SessionScopedRepository:
    """:class:`CardRepository` that runs every call in its own short-lived session.

    SQLAlchemy sessions are not thread-safe and should not outlive a request,
    but the resolver (and therefore the long-lived :class:`RecognitionService`)
    needs a repository object.  This adapter opens a session per call, delegates
    to :class:`SQLAlchemyCardRepository` and closes it again, so concurrent API
    requests never share a session.  The ORM objects it returns are detached
    but every relationship the resolver reads (``card.printings``,
    ``printing.card``) is eagerly loaded by the underlying queries.
    """

    def __init__(self, database: Database) -> None:
        self._database = database

    @property
    def database(self) -> Database:
        return self._database

    def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        with self._database.session() as session:
            return getattr(SQLAlchemyCardRepository(session), method)(*args, **kwargs)

    def get_card(self, card_id: int) -> Card | None:
        return self._call("get_card", card_id)

    def search_cards(self, query: str, *, limit: int = 20) -> Sequence[Card]:
        return self._call("search_cards", query, limit=limit)

    def all_card_names(self) -> Sequence[tuple[int, str]]:
        return self._call("all_card_names")

    def find_printings_by_set_code(self, set_code: str) -> Sequence[Printing]:
        return self._call("find_printings_by_set_code", set_code)

    def find_printings_by_set_codes(self, set_codes: Iterable[str]) -> Sequence[Printing]:
        return self._call("find_printings_by_set_codes", list(set_codes))

    def printings_for_card(self, card_id: int) -> Sequence[Printing]:
        return self._call("printings_for_card", card_id)

    def count_cards(self) -> int:
        return self._call("count_cards")

    def count_printings(self) -> int:
        return self._call("count_printings")


def as_repository(source: RepositorySource) -> CardRepository:
    """Turn a :class:`Database` or a zero-argument factory into a repository."""
    if isinstance(source, Database):
        return SessionScopedRepository(source)
    if callable(source):
        return source()
    raise TypeError(f"expected a Database or a repository factory, got {type(source).__name__}")


def build_recognition_service(
    settings: Settings,
    repository_source: RepositorySource,
    *,
    geometry: CardGeometryService | None = None,
    ocr: OCRProvider | None = None,
    visual: VisualRecognizer | None = None,
    name_index: NameIndex | None = None,
) -> RecognitionService:
    """Build a ready-to-use :class:`RecognitionService`.

    The :class:`NameIndex` is built once here from the repository (pass
    ``name_index`` to reuse one); refresh it after a data sync with
    :meth:`RecognitionService.refresh_name_index`.  The OCR model is loaded
    lazily on first use unless :meth:`RecognitionService.warmup` is called.
    The keyword overrides let tests and tools swap individual stages.
    """
    repository = as_repository(repository_source)
    geometry = geometry or CardGeometryService(settings)
    ocr = ocr or build_ocr_provider(settings)
    visual = visual or build_visual_recognizer(settings.visual_recognizer)
    resolver = Resolver(repository, settings, name_index=name_index)
    logger.info(
        "Recognition service ready: ocr=%s visual=%s layout=%s name_index=%d entries",
        ocr.name, visual.name, settings.card_layout, len(resolver.name_index),
    )
    return RecognitionService(settings, geometry=geometry, ocr=ocr, resolver=resolver, visual=visual)
