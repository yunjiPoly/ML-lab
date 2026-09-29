"""In-memory fuzzy index over normalized card names.

The index is built once from ``CardRepository.all_card_names()`` and queried
with a normalized OCR reading.  Scoring is the maximum of rapidfuzz
``fuzz.ratio`` and ``fuzz.token_sort_ratio`` (scaled to ``0..1``); an exact
normalized match always scores ``1.0``.  ``partial_ratio`` is intentionally not
used because it makes any name that *contains* the query a perfect match
(``"dark magician"`` vs ``"dark magician girl"``).

Results are ranked by score (descending), then by name length (shorter first,
so the plainer name wins a tie against a longer superset name) and finally by
card id for determinism.

The index holds no database session and is safe to keep for the lifetime of the
process; call :meth:`NameIndex.refresh` after a data sync.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

from rapidfuzz import fuzz, process

from app.core.text import normalize_card_name
from app.db.protocols import CardRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NameMatch:
    """One fuzzy hit of :meth:`NameIndex.search`."""

    card_id: int
    normalized_name: str
    score: float

    @property
    def is_exact(self) -> bool:
        return self.score >= 1.0


class NameIndex:
    """Fuzzy name lookup over ``(card_id, normalized_name)`` pairs."""

    def __init__(self, entries: Iterable[tuple[int, str]] = ()) -> None:
        self._card_ids: list[int] = []
        self._names: list[str] = []
        self._load(entries)

    # ----------------------------------------------------------- construction
    @classmethod
    def from_repository(cls, repository: CardRepository) -> "NameIndex":
        """Build the index from every card known to ``repository``."""
        index = cls(repository.all_card_names())
        logger.info("Name index built with %d entries.", len(index))
        return index

    def refresh(self, repository: CardRepository) -> None:
        """Rebuild the index in place (after a data sync)."""
        self._load(repository.all_card_names())
        logger.info("Name index refreshed with %d entries.", len(self))

    def _load(self, entries: Iterable[tuple[int, str]]) -> None:
        card_ids: list[int] = []
        names: list[str] = []
        for card_id, name in entries:
            normalized = normalize_card_name(name)
            if not normalized:
                continue
            card_ids.append(int(card_id))
            names.append(normalized)
        self._card_ids = card_ids
        self._names = names

    # ---------------------------------------------------------------- queries
    def __len__(self) -> int:
        return len(self._names)

    def search(self, query_normalized: str, *, limit: int = 10, score_cutoff: float = 0.0) -> list[NameMatch]:
        """Best ``limit`` names for ``query_normalized`` scoring at least ``score_cutoff``.

        ``query_normalized`` is expected to be the output of
        :func:`normalize_card_name`; normalization is re-applied defensively
        (it is idempotent).  ``score_cutoff`` and returned scores are in ``0..1``.
        """
        query = normalize_card_name(query_normalized)
        if not query or not self._names or limit <= 0:
            return []
        cutoff = max(0.0, min(1.0, score_cutoff)) * 100.0

        best: dict[int, float] = {}
        for scorer in (fuzz.ratio, fuzz.token_sort_ratio):
            hits = process.extract(query, self._names, scorer=scorer, limit=None, score_cutoff=cutoff)
            for _name, score, position in hits:
                if score > best.get(position, -1.0):
                    best[position] = float(score)

        matches: list[NameMatch] = []
        for position, score in best.items():
            name = self._names[position]
            scaled = 1.0 if name == query else min(1.0, score / 100.0)
            matches.append(NameMatch(card_id=self._card_ids[position], normalized_name=name, score=scaled))

        matches.sort(key=lambda m: (-m.score, len(m.normalized_name), m.card_id))
        return matches[:limit]
