"""SQLAlchemy implementation of :class:`app.db.protocols.CardRepository`.

Besides the read side used by the resolver/API, the repository owns the *write*
path used by the sync service: :meth:`SQLAlchemyCardRepository.upsert_card_record`
persists a provider-neutral :class:`~app.services.providers.base.CardRecord`
idempotently.  A content hash of the record is stored on the card so that an
unchanged record is detected without touching the child tables ("do not
re-persist unchanged data").
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload, selectinload

from app.core.text import normalize_card_name, normalize_set_code
from app.models import Artwork, Card, Printing
from app.services.providers.base import ArtworkRecord, CardRecord, PrintingRecord

logger = logging.getLogger(__name__)


class UpsertOutcome(str, Enum):
    """What :meth:`SQLAlchemyCardRepository.upsert_card_record_with_outcome` did."""

    CREATED = "created"
    UPDATED = "updated"
    UNCHANGED = "unchanged"


@dataclass
class UpsertCounts:
    """Aggregated outcomes of a batch upsert."""

    created: int = 0
    updated: int = 0
    unchanged: int = 0

    def add(self, outcome: UpsertOutcome) -> None:
        if outcome is UpsertOutcome.CREATED:
            self.created += 1
        elif outcome is UpsertOutcome.UPDATED:
            self.updated += 1
        else:
            self.unchanged += 1

    @property
    def total(self) -> int:
        return self.created + self.updated + self.unchanged

    @property
    def written(self) -> int:
        """Cards that were actually inserted or updated."""
        return self.created + self.updated


def _clean(value: str | None) -> str | None:
    """Strip a provider string; empty strings become ``None``."""
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def compute_content_hash(record: CardRecord) -> str:
    """Stable SHA-256 of a :class:`CardRecord`.

    Artworks and printings are sorted before hashing so that a provider that
    merely reorders its lists does not trigger a rewrite.
    """
    payload = asdict(record)
    payload["artworks"] = sorted(payload["artworks"], key=lambda a: a["id"])
    payload["printings"] = sorted(
        payload["printings"],
        key=lambda p: (p["set_code"] or "", p["rarity"] or "", p["set_name"] or "", p["rarity_code"] or ""),
    )
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def dedupe_printings(printings: Iterable[PrintingRecord]) -> dict[tuple[str, str | None], PrintingRecord]:
    """Key printings by ``(normalized set code, rarity)``; drop empty codes and duplicates."""
    keyed: dict[tuple[str, str | None], PrintingRecord] = {}
    for printing in printings:
        code = normalize_set_code(printing.set_code)
        if not code:
            logger.debug("Skipping printing without a usable set code: %r", printing)
            continue
        key = (code, _clean(printing.rarity))
        if key in keyed:
            logger.debug("Duplicate printing %s ignored", key)
            continue
        keyed[key] = printing
    return keyed


def dedupe_artworks(artworks: Iterable[ArtworkRecord]) -> dict[int, ArtworkRecord]:
    """Key artworks by id, first occurrence wins."""
    keyed: dict[int, ArtworkRecord] = {}
    for artwork in artworks:
        keyed.setdefault(int(artwork.id), artwork)
    return keyed


class SQLAlchemyCardRepository:
    """Card repository backed by a SQLAlchemy :class:`Session`.

    The repository never commits: the caller owns the transaction (see
    :meth:`app.db.session.Database.session`).
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    @property
    def session(self) -> Session:
        return self._session

    # ------------------------------------------------------------ read side

    def get_card(self, card_id: int) -> Card | None:
        """Card with artworks and printings eagerly loaded, or ``None``."""
        stmt = (
            select(Card)
            .where(Card.id == card_id)
            .options(selectinload(Card.artworks), selectinload(Card.printings))
        )
        return self._session.scalars(stmt).first()

    def search_cards(self, query: str, *, limit: int = 20) -> Sequence[Card]:
        """Case-insensitive search on the normalized name: prefix matches first, then substring matches."""
        needle = normalize_card_name(query)
        if not needle or limit <= 0:
            return []
        prefix_stmt = (
            select(Card)
            .where(Card.normalized_name.like(f"{needle}%"))
            .order_by(Card.name, Card.id)
            .limit(limit)
        )
        results: list[Card] = list(self._session.scalars(prefix_stmt).all())
        remaining = limit - len(results)
        if remaining > 0:
            contains_stmt = (
                select(Card)
                .where(Card.normalized_name.like(f"%{needle}%"))
                .where(~Card.normalized_name.like(f"{needle}%"))
                .order_by(Card.name, Card.id)
                .limit(remaining)
            )
            results.extend(self._session.scalars(contains_stmt).all())
        return results

    def all_card_names(self) -> Sequence[tuple[int, str]]:
        """``(card_id, normalized_name)`` for every card."""
        rows = self._session.execute(select(Card.id, Card.normalized_name).order_by(Card.id)).all()
        return [(int(card_id), name) for card_id, name in rows]

    def find_printings_by_set_code(self, set_code: str) -> Sequence[Printing]:
        """Printings whose normalized set code equals ``set_code`` (normalized before querying)."""
        return self.find_printings_by_set_codes([set_code])

    def find_printings_by_set_codes(self, set_codes: Iterable[str]) -> Sequence[Printing]:
        """Printings whose normalized set code is in ``set_codes``; ``card`` is loaded."""
        codes = sorted({normalize_set_code(code) for code in set_codes} - {""})
        if not codes:
            return []
        stmt = (
            select(Printing)
            .where(Printing.set_code.in_(codes))
            .options(joinedload(Printing.card))
            .order_by(Printing.set_code, Printing.card_id, Printing.rarity)
        )
        return list(self._session.scalars(stmt).all())

    def printings_for_card(self, card_id: int) -> Sequence[Printing]:
        stmt = (
            select(Printing)
            .where(Printing.card_id == card_id)
            .options(joinedload(Printing.card))
            .order_by(Printing.set_code, Printing.rarity)
        )
        return list(self._session.scalars(stmt).all())

    def count_cards(self) -> int:
        return int(self._session.scalar(select(func.count()).select_from(Card)) or 0)

    def count_printings(self) -> int:
        return int(self._session.scalar(select(func.count()).select_from(Printing)) or 0)

    def count_artworks(self) -> int:
        return int(self._session.scalar(select(func.count()).select_from(Artwork)) or 0)

    def count_artworks_with_local_image(self) -> int:
        stmt = select(func.count()).select_from(Artwork).where(Artwork.local_path.is_not(None))
        return int(self._session.scalar(stmt) or 0)

    # ----------------------------------------------------------- write side

    def upsert_card_record(self, record: CardRecord) -> Card:
        """Insert or update a card from a provider record and return the ORM card."""
        card, _ = self.upsert_card_record_with_outcome(record)
        return card

    def upsert_card_records(self, records: Iterable[CardRecord]) -> int:
        """Upsert many records; returns the number of records processed."""
        return self.upsert_card_records_with_counts(records).total

    def upsert_card_records_with_counts(self, records: Iterable[CardRecord]) -> UpsertCounts:
        counts = UpsertCounts()
        for record in records:
            _, outcome = self.upsert_card_record_with_outcome(record)
            counts.add(outcome)
        return counts

    def upsert_card_record_with_outcome(self, record: CardRecord) -> tuple[Card, UpsertOutcome]:
        """Upsert one record and report whether it was created, updated or unchanged.

        Unchanged records (same content hash as stored) are returned without any
        write.  Artworks and printings are reconciled in place, so re-syncing
        never produces duplicate printings.
        """
        content_hash = compute_content_hash(record)
        card = self.get_card(int(record.id))
        if card is None:
            card = Card(id=int(record.id))
            self._session.add(card)
            outcome = UpsertOutcome.CREATED
        elif card.content_hash == content_hash:
            return card, UpsertOutcome.UNCHANGED
        else:
            outcome = UpsertOutcome.UPDATED

        self._apply_scalar_fields(card, record)
        self._reconcile_artworks(card, record)
        self._reconcile_printings(card, record)
        card.content_hash = content_hash
        card.synced_at = datetime.now(timezone.utc)
        self._session.flush()
        logger.debug("Card %s (%s): %s", card.id, card.name, outcome.value)
        return card, outcome

    # --------------------------------------------------------------- helpers

    @staticmethod
    def _apply_scalar_fields(card: Card, record: CardRecord) -> None:
        card.name = record.name.strip()
        card.normalized_name = normalize_card_name(record.name)
        card.type = _clean(record.type)
        card.frame_type = _clean(record.frame_type)
        card.description = record.description
        card.race = _clean(record.race)
        card.attribute = _clean(record.attribute)
        card.level = record.level
        card.atk = record.atk
        card.def_ = record.def_
        card.archetype = _clean(record.archetype)

    def _reconcile_artworks(self, card: Card, record: CardRecord) -> None:
        wanted = dedupe_artworks(record.artworks)
        existing = {artwork.id: artwork for artwork in card.artworks}
        for orphan_id in set(existing) - set(wanted):
            card.artworks.remove(existing.pop(orphan_id))
        self._session.flush()
        for artwork_id, art_record in wanted.items():
            artwork = existing.get(artwork_id)
            if artwork is None:
                # The image id is a provider-wide key; it may already exist on another card row.
                artwork = self._session.get(Artwork, artwork_id) or Artwork(id=artwork_id)
                card.artworks.append(artwork)
            artwork.image_url = _clean(art_record.image_url)
            artwork.image_url_small = _clean(art_record.image_url_small)
            artwork.image_url_cropped = _clean(art_record.image_url_cropped)
            # local_path is owned by the sync service (image cache) and is preserved on update.

    def _reconcile_printings(self, card: Card, record: CardRecord) -> None:
        wanted = dedupe_printings(record.printings)
        existing = {(printing.set_code, _clean(printing.rarity)): printing for printing in card.printings}
        for orphan_key in set(existing) - set(wanted):
            card.printings.remove(existing.pop(orphan_key))
        self._session.flush()
        for (code, rarity), printing_record in wanted.items():
            printing = existing.get((code, rarity))
            if printing is None:
                printing = Printing(set_code=code, rarity=rarity)
                card.printings.append(printing)
            printing.set_name = (printing_record.set_name or "").strip()
            printing.rarity_code = _clean(printing_record.rarity_code)
