"""SyncService with an in-memory fake provider (no network, no real images)."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from app.core.config import Settings
from app.db.repository import SQLAlchemyCardRepository
from app.db.session import Database
from app.models import Artwork
from app.services.providers.base import ArtworkRecord, CardDataProvider, CardRecord
from app.services.providers.ygoprodeck import parse_response
from app.services.sync import DEV_CARD_NAMES, SyncReport, SyncService

ARMADES_ID = 88033975


class FakeProvider(CardDataProvider):
    """Serves records from memory and counts artwork downloads."""

    name = "fake"

    def __init__(self, records: list[CardRecord], *, fail_download_for: set[int] | None = None) -> None:
        self.records = records
        self.download_calls: list[tuple[int, Path]] = []
        self.fail_download_for = fail_download_for or set()

    def fetch_cards_by_names(self, names: Iterable[str]) -> list[CardRecord]:
        wanted = {name.lower() for name in names}
        return [record for record in self.records if record.name.lower() in wanted]

    def iter_all_cards(self) -> Iterator[CardRecord]:
        yield from self.records

    def download_artwork(self, artwork: ArtworkRecord, destination: Path, *, overwrite: bool = False) -> Path | None:
        self.download_calls.append((artwork.id, destination))
        if artwork.id in self.fail_download_for:
            raise OSError("simulated download failure")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"fake-jpeg")
        return destination


@pytest.fixture()
def records(ygoprodeck_sample: dict) -> list[CardRecord]:
    return parse_response(ygoprodeck_sample)


@pytest.fixture()
def db(test_settings: Settings) -> Iterator[Database]:
    database = Database(test_settings.resolved_database_url)
    try:
        yield database
    finally:
        database.dispose()


def test_dev_card_names_match_legacy_list() -> None:
    assert len(DEV_CARD_NAMES) == 11
    assert DEV_CARD_NAMES[0] == "Blue-Eyes White Dragon"
    assert "Armades, Keeper of Boundaries" in DEV_CARD_NAMES


def test_sync_names_without_images(db: Database, test_settings: Settings, records: list[CardRecord]) -> None:
    provider = FakeProvider(records)
    service = SyncService(db, provider, test_settings)

    report = service.sync_names(["Armades, Keeper of Boundaries", "Pot of Greed"], download_images=False)

    assert isinstance(report, SyncReport)
    assert (report.cards_seen, report.cards_created, report.cards_updated, report.cards_unchanged) == (2, 2, 0, 0)
    assert report.printings_total == 5 + 21
    assert report.images_downloaded == 0 and report.images_skipped == 0
    assert provider.download_calls == []
    assert report.errors == []
    assert not test_settings.resolved_card_image_dir.exists()
    with db.session() as session:
        repository = SQLAlchemyCardRepository(session)
        assert repository.count_cards() == 2
        assert repository.get_card(ARMADES_ID).artworks[0].local_path is None


def test_sync_names_reports_unknown_names(db: Database, test_settings: Settings, records: list[CardRecord]) -> None:
    report = SyncService(db, FakeProvider(records), test_settings).sync_names(["Pot of Greed", "Not A Card"], download_images=False)
    assert report.cards_seen == 1
    assert report.errors == ["card not found: 'Not A Card'"]


def test_resync_is_unchanged_then_updated(db: Database, test_settings: Settings, records: list[CardRecord]) -> None:
    provider = FakeProvider(records)
    service = SyncService(db, provider, test_settings)
    first = service.sync_names(DEV_CARD_NAMES, download_images=False)
    assert first.cards_created == 7  # only 7 of the 11 dev names exist in the fixture
    assert len(first.errors) == 4

    second = service.sync_names(DEV_CARD_NAMES, download_images=False)
    assert (second.cards_created, second.cards_updated, second.cards_unchanged) == (0, 0, 7)

    provider.records = [replace(r, atk=1) if r.id == ARMADES_ID else r for r in records]
    third = service.sync_names(["Armades, Keeper of Boundaries"], download_images=False)
    assert (third.cards_created, third.cards_updated, third.cards_unchanged) == (0, 1, 0)
    with db.session() as session:
        assert SQLAlchemyCardRepository(session).count_printings() == sum(len(r.printings) for r in records)


def test_sync_with_images_downloads_once_and_sets_local_path(db: Database, test_settings: Settings, records: list[CardRecord]) -> None:
    provider = FakeProvider(records)
    service = SyncService(db, provider, test_settings)
    image_dir = test_settings.resolved_card_image_dir
    # pre-existing cached file (like the legacy data/cards folder) must be reused, not downloaded
    image_dir.mkdir(parents=True)
    (image_dir / f"{ARMADES_ID}.jpg").write_bytes(b"existing")

    report = service.sync_names(["Armades, Keeper of Boundaries", "Kuriboh"], download_images=True)

    kuriboh = next(r for r in records if r.name == "Kuriboh")
    assert report.images_skipped == 1
    assert report.images_downloaded == len(kuriboh.artworks) == 3
    assert sorted(artwork_id for artwork_id, _ in provider.download_calls) == sorted(a.id for a in kuriboh.artworks)
    assert all(path.parent == image_dir and path.name == f"{artwork_id}.jpg" for artwork_id, path in provider.download_calls)
    with db.session() as session:
        for artwork in session.query(Artwork).all():
            assert artwork.local_path == f"{artwork.id}.jpg"
            assert (image_dir / artwork.local_path).is_file()

    # second run: everything cached -> no download calls
    provider.download_calls.clear()
    again = service.sync_names(["Armades, Keeper of Boundaries", "Kuriboh"], download_images=True)
    assert again.images_downloaded == 0 and again.images_skipped == 4
    assert provider.download_calls == []


def test_image_download_failure_is_recorded_not_fatal(db: Database, test_settings: Settings, records: list[CardRecord]) -> None:
    provider = FakeProvider(records, fail_download_for={ARMADES_ID})
    report = SyncService(db, provider, test_settings).sync_names(["Armades, Keeper of Boundaries"], download_images=True)
    assert report.cards_created == 1
    assert report.images_downloaded == 0
    assert len(report.errors) == 1 and "simulated download failure" in report.errors[0]


def test_sync_all_batches_and_limit(db: Database, test_settings: Settings, records: list[CardRecord]) -> None:
    service = SyncService(db, FakeProvider(records), test_settings, batch_size=2)
    report = service.sync_all(download_images=False, limit=5)
    assert (report.cards_seen, report.cards_created) == (5, 5)
    with db.session() as session:
        assert SQLAlchemyCardRepository(session).count_cards() == 5
    full = service.sync_all(download_images=False)
    assert (full.cards_created, full.cards_unchanged) == (2, 5)


def test_bad_record_isolated_from_batch(db: Database, test_settings: Settings, records: list[CardRecord]) -> None:
    broken = CardRecord(id=999, name=None)  # type: ignore[arg-type]  # name is NOT NULL -> fails on flush
    service = SyncService(db, FakeProvider(records[:3] + [broken] + records[3:]), test_settings, batch_size=10)
    report = service.sync_all(download_images=False)
    assert report.cards_seen == 8
    assert report.cards_created == 7
    assert len(report.errors) == 1 and "999" in report.errors[0]
    with db.session() as session:
        assert SQLAlchemyCardRepository(session).count_cards() == 7


def test_provider_failure_is_reported(db: Database, test_settings: Settings) -> None:
    class ExplodingProvider(FakeProvider):
        def fetch_cards_by_names(self, names: Iterable[str]) -> list[CardRecord]:
            raise RuntimeError("network down")

    report = SyncService(db, ExplodingProvider([]), test_settings).sync_names(["Kuriboh"], download_images=False)
    assert report.cards_seen == 0
    assert report.errors == ["provider error: network down"]
    assert "errors:" in report.summary()
