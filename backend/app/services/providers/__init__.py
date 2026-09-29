"""External data providers (card metadata, market prices).

The application must never depend on a single provider's quirks outside this
package.  Licensing / terms of every provider must be reviewed before any
commercial deployment.

Concrete providers (:class:`YGOProDeckProvider`, :class:`JsonFileProvider`) are
imported lazily from their modules to keep this package import light.
"""

from app.services.providers.base import (
    ArtworkRecord,
    CardDataProvider,
    CardRecord,
    MarketPriceProvider,
    PrintingRecord,
)

__all__ = [
    "ArtworkRecord",
    "CardDataProvider",
    "CardRecord",
    "MarketPriceProvider",
    "PrintingRecord",
    "JsonFileProvider",
    "YGOProDeckProvider",
]


def __getattr__(name: str):  # pragma: no cover - trivial lazy import
    if name == "YGOProDeckProvider":
        from app.services.providers.ygoprodeck import YGOProDeckProvider

        return YGOProDeckProvider
    if name == "JsonFileProvider":
        from app.services.providers.json_file import JsonFileProvider

        return JsonFileProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
