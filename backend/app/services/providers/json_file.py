"""Offline card provider backed by a JSON file.

Two layouts are accepted and auto-detected per entry:

* a raw YGOPRODeck API dump ``{"data": [...]}`` (or a bare list of API card
  objects), parsed with :func:`app.services.providers.ygoprodeck.parse_card`;
* the legacy list written by ``legacy/download_cards.py`` (``data/cards.json``)
  with ``frame_type`` / ``description`` / ``artworks[{id,image,image_url}]`` /
  ``printings[{set_name,set_code,rarity,rarity_code}]``.

Legacy entries may reference an already-downloaded image (``artworks[].image``);
:meth:`JsonFileProvider.download_artwork` copies that file instead of hitting
the network when it exists.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import requests

from app.core.text import normalize_card_name
from app.services.providers.base import ArtworkRecord, CardDataProvider, CardRecord, PrintingRecord
from app.services.providers.ygoprodeck import RetryingHttpClient, download_to_path, parse_card

logger = logging.getLogger(__name__)

_API_MARKERS = ("card_sets", "card_images", "frameType", "desc")


def is_api_shaped(item: dict) -> bool:
    """True when a card object uses the raw YGOPRODeck field names."""
    return any(marker in item for marker in _API_MARKERS)


def _opt_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def parse_legacy_card(payload: dict) -> tuple[CardRecord, dict[int, str]]:
    """Parse one legacy ``data/cards.json`` entry.

    Returns the record plus a mapping ``artwork id -> local image path`` (as
    written in the file, possibly with Windows separators) for artworks that
    carried an ``image`` field.
    """
    artworks: list[ArtworkRecord] = []
    local_images: dict[int, str] = {}
    for image in payload.get("artworks") or []:
        if image.get("id") is None:
            continue
        artwork_id = int(image["id"])
        artworks.append(ArtworkRecord(id=artwork_id, image_url=_opt_str(image.get("image_url"))))
        local = _opt_str(image.get("image"))
        if local:
            local_images[artwork_id] = local
    printings = [
        PrintingRecord(
            set_code=str(printing.get("set_code") or "").strip(),
            set_name=str(printing.get("set_name") or "").strip(),
            rarity=_opt_str(printing.get("rarity")),
            rarity_code=_opt_str(printing.get("rarity_code")),
        )
        for printing in payload.get("printings") or []
        if printing.get("set_code")
    ]
    record = CardRecord(
        id=int(payload["id"]),
        name=str(payload["name"]).strip(),
        type=_opt_str(payload.get("type")),
        frame_type=_opt_str(payload.get("frame_type")),
        description=payload.get("description"),
        race=_opt_str(payload.get("race")),
        attribute=_opt_str(payload.get("attribute")),
        level=payload.get("level"),
        atk=payload.get("atk"),
        def_=payload.get("def_", payload.get("def")),
        archetype=_opt_str(payload.get("archetype")),
        artworks=artworks,
        printings=printings,
    )
    return record, local_images


class JsonFileProvider(CardDataProvider):
    """Card provider reading a local JSON catalog (no network needed for metadata)."""

    name = "json_file"

    def __init__(
        self,
        path: Path,
        *,
        session: requests.Session | None = None,
        request_delay_seconds: float = 0.15,
        timeout_seconds: float = 60.0,
        http_client_factory: Callable[[], RetryingHttpClient] | None = None,
    ) -> None:
        self.path = Path(path)
        self._records: list[CardRecord] | None = None
        self._local_images: dict[int, Path] = {}
        self._session = session
        self._request_delay = request_delay_seconds
        self._timeout = timeout_seconds
        self._http_client_factory = http_client_factory
        self._http_client: RetryingHttpClient | None = None

    # --------------------------------------------------------------- loading

    def _load(self) -> list[CardRecord]:
        if self._records is not None:
            return self._records
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        items = (payload.get("data") or []) if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            raise ValueError(f"{self.path}: expected a list of cards or an object with a 'data' list")
        records: list[CardRecord] = []
        for item in items:
            if not isinstance(item, dict):
                logger.warning("%s: skipping non-object entry %r", self.path, item)
                continue
            try:
                if is_api_shaped(item):
                    records.append(parse_card(item))
                else:
                    record, local_images = parse_legacy_card(item)
                    records.append(record)
                    for artwork_id, raw_path in local_images.items():
                        resolved = self._resolve_local_image(raw_path)
                        if resolved is not None:
                            self._local_images[artwork_id] = resolved
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("%s: skipping malformed entry (%s): %r", self.path, exc, item.get("id"))
        self._records = records
        logger.info("Loaded %d cards from %s", len(records), self.path)
        return records

    def _resolve_local_image(self, raw_path: str) -> Path | None:
        """Find a legacy image path: as written, relative to the JSON file, its parent, or the cwd.

        The JSON file's own directories are tried before the current working
        directory so a catalog is self-contained wherever the process runs.
        """
        candidate = Path(raw_path.replace("\\", "/"))
        bases = [self.path.parent, self.path.parent.parent, Path.cwd()]
        options = [candidate] if candidate.is_absolute() else [base / candidate for base in bases]
        for option in options:
            if option.is_file():
                return option.resolve()
        return None

    @property
    def records(self) -> list[CardRecord]:
        """All records in the file (loaded lazily on first access)."""
        return list(self._load())

    def local_image_path(self, artwork_id: int) -> Path | None:
        """Existing local image referenced by the file for ``artwork_id``, if any."""
        self._load()
        return self._local_images.get(int(artwork_id))

    # -------------------------------------------------------------- provider

    def fetch_cards_by_names(self, names: Iterable[str]) -> list[CardRecord]:
        """Records whose name matches one of ``names`` (exact after normalization), file order."""
        wanted = {normalize_card_name(name) for name in names if name} - {""}
        return [record for record in self._load() if normalize_card_name(record.name) in wanted]

    def iter_all_cards(self) -> Iterator[CardRecord]:
        yield from self._load()

    def _client(self) -> RetryingHttpClient:
        if self._http_client is None:
            if self._http_client_factory is not None:
                self._http_client = self._http_client_factory()
            else:
                self._http_client = RetryingHttpClient(
                    session=self._session,
                    request_delay_seconds=self._request_delay,
                    timeout_seconds=self._timeout,
                )
        return self._http_client

    def download_artwork(self, artwork: ArtworkRecord, destination: Path, *, overwrite: bool = False) -> Path | None:
        """Copy the referenced local image when present, otherwise download ``image_url``."""
        destination = Path(destination)
        if destination.exists() and not overwrite:
            return destination
        local = self.local_image_path(artwork.id)
        if local is not None and local.is_file():
            if local.resolve() == destination.resolve():
                return destination
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local, destination)
            logger.debug("Copied local artwork %s -> %s", local, destination)
            return destination
        if not artwork.image_url:
            logger.warning("Artwork %s has neither a local image nor an image_url", artwork.id)
            return None
        return download_to_path(self._client(), artwork.image_url, destination, overwrite=overwrite)
