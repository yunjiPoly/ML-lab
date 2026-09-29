"""In-memory :class:`app.db.protocols.CardRepository` for unit tests.

Builds plain ORM instances (``Card`` / ``Printing`` / ``Artwork``) directly from
the raw YGOPRODeck-shaped fixture ``ygoprodeck_sample.json`` WITHOUT a database
session.  Values are stored exactly as the real repository would store them:
``Card.normalized_name`` via :func:`normalize_card_name` and
``Printing.set_code`` via :func:`normalize_set_code`.

Usage::

    repo = FakeCardRepository.from_sample()                # the 7 fixture cards
    repo = FakeCardRepository.from_payload(ygoprodeck_sample)
    repo.add_card(900001, "Harpie Lady 1", printings=[("LOB-EN049", "Legend of Blue Eyes", "Common", "C")])
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

from app.core.text import normalize_card_name, normalize_set_code
from app.models import Artwork, Card, Printing

FIXTURES_DIR = Path(__file__).resolve().parent
SAMPLE_PATH = FIXTURES_DIR / "ygoprodeck_sample.json"

# (set_code, set_name, rarity, rarity_code)
PrintingSpec = tuple[str, str, str | None, str | None]


def _clean_rarity_code(value: str | None) -> str | None:
    """``"(ScR)"`` -> ``"ScR"``; empty -> None."""
    if not value:
        return None
    cleaned = value.strip().strip("()").strip()
    return cleaned or None


class FakeCardRepository:
    """Dictionary-backed repository satisfying the ``CardRepository`` protocol."""

    def __init__(self, cards: Iterable[Card] = ()) -> None:
        self._cards: dict[int, Card] = {}
        self._next_printing_id = 1
        for card in cards:
            self.add(card)

    # ------------------------------------------------------------ builders
    @classmethod
    def from_payload(cls, payload: dict) -> "FakeCardRepository":
        """Build from a raw YGOPRODeck API response ``{"data": [...]}`` (or a bare list)."""
        rows = payload.get("data", []) if isinstance(payload, dict) else list(payload)
        repo = cls()
        for row in rows:
            repo.add(build_card(row))
        return repo

    @classmethod
    def from_sample(cls, path: Path = SAMPLE_PATH) -> "FakeCardRepository":
        """Build from ``tests/fixtures/ygoprodeck_sample.json`` (or another dump)."""
        return cls.from_payload(json.loads(path.read_text(encoding="utf-8")))

    def add(self, card: Card) -> Card:
        """Register ``card``; normalizes names/codes, assigns printing ids, drops duplicate printings."""
        card.normalized_name = normalize_card_name(card.name)
        unique: dict[tuple[str, str | None], Printing] = {}
        for printing in list(card.printings):
            printing.set_code = normalize_set_code(printing.set_code)
            key = (printing.set_code, printing.rarity)
            if key in unique or not printing.set_code:
                continue
            if printing.id is None:
                printing.id = self._next_printing_id
                self._next_printing_id += 1
            else:
                self._next_printing_id = max(self._next_printing_id, printing.id + 1)
            printing.card_id = card.id
            unique[key] = printing
        card.printings = sorted(unique.values(), key=lambda p: (p.set_code, p.id or 0))
        for artwork in card.artworks:
            artwork.card_id = card.id
        self._cards[card.id] = card
        return card

    def add_card(
        self,
        card_id: int,
        name: str,
        *,
        type: str | None = None,
        frame_type: str | None = None,
        printings: Sequence[PrintingSpec] = (),
    ) -> Card:
        """Convenience constructor for synthetic test cards."""
        card = Card(id=card_id, name=name, normalized_name=normalize_card_name(name), type=type, frame_type=frame_type)
        card.printings = [
            Printing(card_id=card_id, set_code=code, set_name=set_name, rarity=rarity, rarity_code=rarity_code)
            for code, set_name, rarity, rarity_code in printings
        ]
        return self.add(card)

    # ------------------------------------------------- CardRepository protocol
    def get_card(self, card_id: int) -> Card | None:
        return self._cards.get(card_id)

    def search_cards(self, query: str, *, limit: int = 20) -> list[Card]:
        needle = normalize_card_name(query)
        if not needle:
            return []
        hits = [c for c in self._cards.values() if needle in c.normalized_name]
        hits.sort(key=lambda c: (not c.normalized_name.startswith(needle), c.normalized_name))
        return hits[:limit]

    def all_card_names(self) -> list[tuple[int, str]]:
        return [(card.id, card.normalized_name) for card in self._cards.values()]

    def find_printings_by_set_code(self, set_code: str) -> list[Printing]:
        return self.find_printings_by_set_codes([set_code])

    def find_printings_by_set_codes(self, set_codes: Iterable[str]) -> list[Printing]:
        wanted = {normalize_set_code(code) for code in set_codes}
        wanted.discard("")
        found = [p for card in self._cards.values() for p in card.printings if p.set_code in wanted]
        found.sort(key=lambda p: (p.set_code, p.card_id, p.id or 0))
        return found

    def printings_for_card(self, card_id: int) -> list[Printing]:
        card = self._cards.get(card_id)
        return list(card.printings) if card else []

    def count_cards(self) -> int:
        return len(self._cards)

    def count_printings(self) -> int:
        return sum(len(card.printings) for card in self._cards.values())


def build_card(row: dict) -> Card:
    """Construct a transient ``Card`` (with printings and artworks) from one raw API card."""
    card = Card(
        id=int(row["id"]),
        name=row["name"],
        normalized_name=normalize_card_name(row["name"]),
        type=row.get("type"),
        frame_type=row.get("frameType"),
        description=row.get("desc"),
        race=row.get("race"),
        attribute=row.get("attribute"),
        level=row.get("level"),
        atk=row.get("atk"),
        def_=row.get("def"),
        archetype=row.get("archetype"),
        provider="ygoprodeck",
    )
    card.printings = [
        Printing(
            card_id=card.id,
            set_code=normalize_set_code(entry.get("set_code")),
            set_name=entry.get("set_name") or "",
            rarity=entry.get("set_rarity") or None,
            rarity_code=_clean_rarity_code(entry.get("set_rarity_code")),
        )
        for entry in row.get("card_sets", [])
    ]
    card.artworks = [
        Artwork(
            id=int(image["id"]),
            card_id=card.id,
            image_url=image.get("image_url"),
            image_url_small=image.get("image_url_small"),
            image_url_cropped=image.get("image_url_cropped"),
        )
        for image in row.get("card_images", [])
    ]
    return card
