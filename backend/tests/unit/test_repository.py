"""SQLAlchemyCardRepository: upsert semantics + read queries on an in-memory SQLite."""

from __future__ import annotations

from dataclasses import replace

import pytest
from sqlalchemy import select

from app.db.repository import SQLAlchemyCardRepository, UpsertOutcome, compute_content_hash
from app.db.session import Database
from app.models import Artwork, Card, Printing
from app.services.providers.base import ArtworkRecord, CardRecord, PrintingRecord
from app.services.providers.ygoprodeck import parse_response

ARMADES_ID = 88033975


@pytest.fixture()
def records(ygoprodeck_sample: dict) -> list[CardRecord]:
    return parse_response(ygoprodeck_sample)


@pytest.fixture()
def armades(records: list[CardRecord]) -> CardRecord:
    return next(record for record in records if record.id == ARMADES_ID)


def _upsert(db: Database, record: CardRecord) -> UpsertOutcome:
    with db.session() as session:
        _, outcome = SQLAlchemyCardRepository(session).upsert_card_record_with_outcome(record)
    return outcome


def test_upsert_creates_card_with_children(memory_db: Database, armades: CardRecord) -> None:
    assert _upsert(memory_db, armades) is UpsertOutcome.CREATED
    with memory_db.session() as session:
        card = SQLAlchemyCardRepository(session).get_card(ARMADES_ID)
        assert card is not None
        assert card.name == "Armades, Keeper of Boundaries"
        assert card.normalized_name == "armades keeper of boundaries"
        assert card.def_ == 1500 and card.atk == 2300 and card.level == 5
        assert card.content_hash == compute_content_hash(armades)
        assert card.synced_at is not None
        assert [a.id for a in card.artworks] == [ARMADES_ID]
        assert len(card.printings) == 5
        by_key = {(p.set_code, p.rarity): p for p in card.printings}
        assert by_key[("JOTL-EN045", "Secret Rare")].set_name == "Judgment of the Light"
        assert by_key[("JOTL-EN045", "Secret Rare")].rarity_code == "(ScR)"
        assert ("BLGG-EN090", "Secret Rare") in by_key and ("BLGG-EN090", "Starlight Rare") in by_key


def test_reupsert_unchanged_record_is_a_noop(memory_db: Database, armades: CardRecord) -> None:
    _upsert(memory_db, armades)
    with memory_db.session() as session:
        synced_before = session.get(Card, ARMADES_ID).synced_at
    assert _upsert(memory_db, armades) is UpsertOutcome.UNCHANGED
    with memory_db.session() as session:
        card = SQLAlchemyCardRepository(session).get_card(ARMADES_ID)
        assert card.synced_at == synced_before
        assert len(card.printings) == 5


def test_reupsert_modified_record_updates_in_place(memory_db: Database, armades: CardRecord) -> None:
    _upsert(memory_db, armades)
    modified = replace(armades, atk=9999, printings=armades.printings + [PrintingRecord("NEW-EN001", "New Set", "Common", "(C)")])
    assert _upsert(memory_db, modified) is UpsertOutcome.UPDATED
    with memory_db.session() as session:
        card = SQLAlchemyCardRepository(session).get_card(ARMADES_ID)
        assert card.atk == 9999
        assert len(card.printings) == 6
        assert card.content_hash == compute_content_hash(modified)
        # and dropping a printing removes exactly that row
    reduced = replace(armades, printings=[p for p in armades.printings if p.set_code != "MP14-EN095"])
    assert _upsert(memory_db, reduced) is UpsertOutcome.UPDATED
    with memory_db.session() as session:
        codes = {p.set_code for p in SQLAlchemyCardRepository(session).printings_for_card(ARMADES_ID)}
        assert codes == {"BLGG-EN090", "JOTL-EN045", "PGL2-EN043"}
        assert session.scalar(select(Printing).where(Printing.set_code == "MP14-EN095")) is None


def test_printings_are_deduplicated_and_normalized(memory_db: Database) -> None:
    record = CardRecord(
        id=1,
        name="Dupe Test",
        printings=[
            PrintingRecord("jotl-en045", "Judgment of the Light", "Secret Rare", "(ScR)"),
            PrintingRecord("JOTL–EN045 ", "Judgment of the Light", "Secret Rare", "(ScR)"),  # en dash + space
            PrintingRecord("JOTL-EN045", "Judgment of the Light", "Ultra Rare", "(UR)"),
            PrintingRecord("", "Broken", None, None),
        ],
    )
    _upsert(memory_db, record)
    _upsert(memory_db, replace(record, description="touch"))  # forces a re-reconcile
    with memory_db.session() as session:
        printings = SQLAlchemyCardRepository(session).printings_for_card(1)
        assert sorted((p.set_code, p.rarity) for p in printings) == [
            ("JOTL-EN045", "Secret Rare"),
            ("JOTL-EN045", "Ultra Rare"),
        ]


def test_artworks_preserve_local_path_on_update(memory_db: Database) -> None:
    record = CardRecord(id=5, name="Art", artworks=[ArtworkRecord(50, "https://img/50.jpg"), ArtworkRecord(51, "https://img/51.jpg")])
    _upsert(memory_db, record)
    with memory_db.session() as session:
        session.get(Artwork, 50).local_path = "50.jpg"
    updated = replace(record, artworks=[ArtworkRecord(50, "https://img/50-v2.jpg")], description="changed")
    assert _upsert(memory_db, updated) is UpsertOutcome.UPDATED
    with memory_db.session() as session:
        card = SQLAlchemyCardRepository(session).get_card(5)
        assert [(a.id, a.image_url, a.local_path) for a in card.artworks] == [(50, "https://img/50-v2.jpg", "50.jpg")]
        assert session.get(Artwork, 51) is None


def test_find_printings_by_set_code_normalizes_query(memory_db: Database, records: list[CardRecord]) -> None:
    with memory_db.session() as session:
        SQLAlchemyCardRepository(session).upsert_card_records(records)
    with memory_db.session() as session:
        repository = SQLAlchemyCardRepository(session)
        printings = repository.find_printings_by_set_code("jotl-en045 ")
        assert len(printings) == 1
        assert printings[0].card.name == "Armades, Keeper of Boundaries"
        assert printings[0].rarity == "Secret Rare"
        many = repository.find_printings_by_set_codes(["blgg-en090", "JOTL-EN045", "", "nope-en999"])
        assert sorted((p.set_code, p.rarity) for p in many) == [
            ("BLGG-EN090", "Secret Rare"),
            ("BLGG-EN090", "Starlight Rare"),
            ("JOTL-EN045", "Secret Rare"),
        ]
        assert repository.find_printings_by_set_codes([]) == []


def test_search_cards_prefix_before_contains(memory_db: Database, records: list[CardRecord]) -> None:
    with memory_db.session() as session:
        SQLAlchemyCardRepository(session).upsert_card_records(records)
    with memory_db.session() as session:
        repository = SQLAlchemyCardRepository(session)
        names = [card.name for card in repository.search_cards("Dark Magician")]
        assert names == ["Dark Magician", "Dark Magician Girl"]
        # "magician" is a substring of both, ordered by name
        assert [c.name for c in repository.search_cards("MAGICIAN")] == ["Dark Magician", "Dark Magician Girl"]
        assert [c.name for c in repository.search_cards("magician", limit=1)] == ["Dark Magician"]
        assert [c.name for c in repository.search_cards("Armades, Keeper")] == ["Armades, Keeper of Boundaries"]
        assert repository.search_cards("") == []
        assert repository.search_cards("zzz-not-there") == []


def test_all_card_names_and_counts(memory_db: Database, records: list[CardRecord]) -> None:
    with memory_db.session() as session:
        assert SQLAlchemyCardRepository(session).upsert_card_records(records) == 7
    with memory_db.session() as session:
        repository = SQLAlchemyCardRepository(session)
        names = repository.all_card_names()
        assert len(names) == 7
        assert (ARMADES_ID, "armades keeper of boundaries") in names
        assert all(isinstance(card_id, int) and isinstance(name, str) for card_id, name in names)
        assert repository.count_cards() == 7
        assert repository.count_printings() == sum(len(r.printings) for r in records)
        assert repository.count_artworks() == sum(len(r.artworks) for r in records)
        assert repository.count_artworks_with_local_image() == 0
        assert repository.get_card(123456789) is None


def test_upsert_counts(memory_db: Database, records: list[CardRecord]) -> None:
    with memory_db.session() as session:
        counts = SQLAlchemyCardRepository(session).upsert_card_records_with_counts(records)
        assert (counts.created, counts.updated, counts.unchanged) == (7, 0, 0)
    with memory_db.session() as session:
        counts = SQLAlchemyCardRepository(session).upsert_card_records_with_counts(records)
        assert (counts.created, counts.updated, counts.unchanged) == (0, 0, 7)
        assert counts.total == 7 and counts.written == 0


def test_content_hash_ignores_child_order() -> None:
    record = CardRecord(id=1, name="A", printings=[PrintingRecord("X-1", "S"), PrintingRecord("Y-2", "T")])
    reordered = replace(record, printings=list(reversed(record.printings)))
    assert compute_content_hash(record) == compute_content_hash(reordered)
    assert compute_content_hash(record) != compute_content_hash(replace(record, name="B"))
