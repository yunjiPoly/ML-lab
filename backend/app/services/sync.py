"""Card catalog synchronisation: provider -> database (+ optional image cache).

Records are persisted in batches, each in its own transaction, so a full
catalog sync (~13k cards) never holds one huge transaction.  Errors are
collected in the :class:`SyncReport` instead of aborting the run.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.text import normalize_card_name
from app.db.repository import SQLAlchemyCardRepository, UpsertOutcome
from app.db.session import Database
from app.models import Card
from app.services.providers.base import ArtworkRecord, CardDataProvider, CardRecord

logger = logging.getLogger(__name__)

#: The development subset (same list as ``legacy/download_cards.py``).
DEV_CARD_NAMES: list[str] = [
    "Blue-Eyes White Dragon",
    "Dark Magician",
    "Dark Magician Girl",
    "Red-Eyes Black Dragon",
    "Exodia the Forbidden One",
    "Kuriboh",
    "Monster Reborn",
    "Pot of Greed",
    "Mirror Force",
    "Polymerization",
    "Armades, Keeper of Boundaries",
]

DEFAULT_BATCH_SIZE = 200
IMAGE_SUFFIX = ".jpg"


@dataclass
class SyncReport:
    """Outcome of one sync run."""

    cards_seen: int = 0
    cards_created: int = 0
    cards_updated: int = 0
    cards_unchanged: int = 0
    printings_total: int = 0
    images_downloaded: int = 0
    images_skipped: int = 0
    errors: list[str] = field(default_factory=list)

    def merge(self, other: SyncReport) -> None:
        self.cards_seen += other.cards_seen
        self.cards_created += other.cards_created
        self.cards_updated += other.cards_updated
        self.cards_unchanged += other.cards_unchanged
        self.printings_total += other.printings_total
        self.images_downloaded += other.images_downloaded
        self.images_skipped += other.images_skipped
        self.errors.extend(other.errors)

    def summary(self) -> str:
        """Human readable multi-line summary."""
        lines = [
            f"cards seen:        {self.cards_seen}",
            f"  created:         {self.cards_created}",
            f"  updated:         {self.cards_updated}",
            f"  unchanged:       {self.cards_unchanged}",
            f"printings total:   {self.printings_total}",
            f"images downloaded: {self.images_downloaded}",
            f"images skipped:    {self.images_skipped} (already cached)",
            f"errors:            {len(self.errors)}",
        ]
        lines.extend(f"  - {error}" for error in self.errors[:20])
        if len(self.errors) > 20:
            lines.append(f"  ... {len(self.errors) - 20} more")
        return "\n".join(lines)


def _batched(records: Iterable[CardRecord], size: int) -> Iterable[list[CardRecord]]:
    iterator = iter(records)
    while batch := list(islice(iterator, size)):
        yield batch


class SyncService:
    """Fetches cards from a :class:`CardDataProvider` and upserts them into the database."""

    def __init__(
        self,
        database: Database,
        provider: CardDataProvider,
        settings: Settings,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self._database = database
        self._provider = provider
        self._settings = settings
        self._batch_size = max(1, batch_size)

    @property
    def image_dir(self) -> Path:
        return self._settings.resolved_card_image_dir

    # ------------------------------------------------------------ public api

    def sync_names(self, names: Sequence[str], *, download_images: bool = True) -> SyncReport:
        """Sync the given card names (development mode).  Unknown names are reported as errors."""
        self._database.create_all()
        report = SyncReport()
        try:
            records = self._provider.fetch_cards_by_names(names)
        except Exception as exc:  # provider failure: nothing to persist
            logger.exception("Provider %s failed to fetch cards by name", self._provider.name)
            report.errors.append(f"provider error: {exc}")
            return report
        found = {normalize_card_name(record.name) for record in records}
        for name in names:
            if normalize_card_name(name) not in found:
                message = f"card not found: {name!r}"
                logger.warning(message)
                report.errors.append(message)
        report.merge(self._sync_records(records, download_images=download_images))
        return report

    def sync_all(self, *, download_images: bool = False, limit: int | None = None) -> SyncReport:
        """Sync the complete catalog (optionally only the first ``limit`` cards)."""
        self._database.create_all()
        records: Iterable[CardRecord] = self._provider.iter_all_cards()
        if limit is not None:
            records = islice(records, max(0, limit))
        return self._sync_records(records, download_images=download_images)

    # -------------------------------------------------------------- batches

    def _sync_records(self, records: Iterable[CardRecord], *, download_images: bool) -> SyncReport:
        report = SyncReport()
        if download_images:
            self.image_dir.mkdir(parents=True, exist_ok=True)
        for index, batch in enumerate(_batched(records, self._batch_size), start=1):
            batch_report = self._persist_batch(batch, download_images=download_images)
            report.merge(batch_report)
            logger.info(
                "Batch %d: %d records (created=%d updated=%d unchanged=%d errors=%d, total seen=%d)",
                index,
                len(batch),
                batch_report.cards_created,
                batch_report.cards_updated,
                batch_report.cards_unchanged,
                len(batch_report.errors),
                report.cards_seen,
            )
        return report

    def _persist_batch(self, batch: list[CardRecord], *, download_images: bool) -> SyncReport:
        """Persist one batch in a single transaction; on failure, retry record by record."""
        report = SyncReport()
        try:
            with self._database.session() as session:
                self._upsert_records(session, batch, report, download_images=download_images)
            return report
        except Exception as exc:
            logger.warning("Batch of %d records failed (%s); retrying records individually", len(batch), exc)
        report = SyncReport()
        for record in batch:
            single = SyncReport()
            try:
                with self._database.session() as session:
                    self._upsert_records(session, [record], single, download_images=download_images)
                report.merge(single)
            except Exception as exc:
                logger.exception("Failed to persist card %s (%r)", record.id, record.name)
                report.cards_seen += 1
                report.errors.append(f"card {record.id} ({record.name!r}): {exc}")
        return report

    def _upsert_records(
        self, session: Session, records: list[CardRecord], report: SyncReport, *, download_images: bool
    ) -> None:
        repository = SQLAlchemyCardRepository(session)
        for record in records:
            card, outcome = repository.upsert_card_record_with_outcome(record)
            report.cards_seen += 1
            if outcome is UpsertOutcome.CREATED:
                report.cards_created += 1
            elif outcome is UpsertOutcome.UPDATED:
                report.cards_updated += 1
            else:
                report.cards_unchanged += 1
            report.printings_total += len(card.printings)
            if download_images:
                self._cache_images(card, record, report)

    # --------------------------------------------------------------- images

    def _cache_images(self, card: Card, record: CardRecord, report: SyncReport) -> None:
        """Download (or reuse) every artwork image of ``card`` and store the relative file name."""
        by_id: dict[int, ArtworkRecord] = {int(artwork.id): artwork for artwork in record.artworks}
        for artwork in card.artworks:
            destination = self.image_dir / f"{artwork.id}{IMAGE_SUFFIX}"
            if destination.is_file():
                report.images_skipped += 1
                if artwork.local_path != destination.name:
                    artwork.local_path = destination.name
                continue
            artwork_record = by_id.get(artwork.id)
            if artwork_record is None:
                report.errors.append(f"artwork {artwork.id} of card {card.id}: not present in provider record")
                continue
            try:
                path = self._provider.download_artwork(artwork_record, destination)
            except Exception as exc:  # network / disk problems must not abort the sync
                logger.warning("Artwork %s download failed: %s", artwork.id, exc)
                report.errors.append(f"artwork {artwork.id} of card {card.id}: {exc}")
                continue
            if path is None:
                report.errors.append(f"artwork {artwork.id} of card {card.id}: no image available")
                continue
            artwork.local_path = Path(path).name
            report.images_downloaded += 1
