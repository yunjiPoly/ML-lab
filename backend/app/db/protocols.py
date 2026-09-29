"""Repository protocol used by the resolver.

The resolver only depends on this protocol, so it can be unit-tested with an
in-memory fake and later backed by PostgreSQL without changes.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol

from app.models import Card, Printing


class CardRepository(Protocol):
    def get_card(self, card_id: int) -> Card | None:
        """Card with artworks and printings loaded, or None."""

    def search_cards(self, query: str, *, limit: int = 20) -> Sequence[Card]:
        """Substring / prefix search on the normalized name for the API search endpoint."""

    def all_card_names(self) -> Sequence[tuple[int, str]]:
        """``(card_id, normalized_name)`` for every card; used to build the fuzzy index."""

    def find_printings_by_set_code(self, set_code: str) -> Sequence[Printing]:
        """Printings whose normalized set code equals ``set_code`` (with ``card`` loaded)."""

    def find_printings_by_set_codes(self, set_codes: Iterable[str]) -> Sequence[Printing]:
        """Printings whose normalized set code is in ``set_codes`` (with ``card`` loaded)."""

    def printings_for_card(self, card_id: int) -> Sequence[Printing]:
        """All printings of a card."""

    def count_cards(self) -> int: ...

    def count_printings(self) -> int: ...
