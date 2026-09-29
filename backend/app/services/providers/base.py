"""Provider interfaces + provider-neutral records."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ArtworkRecord:
    id: int
    image_url: str | None = None
    image_url_small: str | None = None
    image_url_cropped: str | None = None


@dataclass(frozen=True)
class PrintingRecord:
    set_code: str  # raw provider value; normalized by the repository on persist
    set_name: str
    rarity: str | None = None
    rarity_code: str | None = None


@dataclass
class CardRecord:
    """Provider-neutral card with its artworks and printings."""

    id: int
    name: str
    type: str | None = None
    frame_type: str | None = None
    description: str | None = None
    race: str | None = None
    attribute: str | None = None
    level: int | None = None
    atk: int | None = None
    def_: int | None = None
    archetype: str | None = None
    artworks: list[ArtworkRecord] = field(default_factory=list)
    printings: list[PrintingRecord] = field(default_factory=list)


class CardDataProvider(ABC):
    """Source of card metadata (YGOPRODeck today)."""

    name: str = "base"

    @abstractmethod
    def fetch_cards_by_names(self, names: Iterable[str]) -> list[CardRecord]:
        """Fetch specific cards by exact name (development mode)."""

    @abstractmethod
    def iter_all_cards(self) -> Iterator[CardRecord]:
        """Stream the complete catalog (full mode)."""

    @abstractmethod
    def download_artwork(self, artwork: ArtworkRecord, destination: Path, *, overwrite: bool = False) -> Path | None:
        """Download an artwork image to ``destination`` unless it already exists.  Returns the path or None."""


class MarketPriceProvider(ABC):
    """Placeholder for TCGplayer / Cardmarket integrations.  Not required by the MVP.

    Nothing in the application may *require* credentials for these providers.
    """

    name: str = "base"

    @abstractmethod
    def get_prices(self, set_code: str) -> dict[str, float]:
        """Return a mapping of price label -> value for a printing."""
