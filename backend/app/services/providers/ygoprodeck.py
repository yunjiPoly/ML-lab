"""YGOPRODeck card database provider (https://ygoprodeck.com/api-guide/).

Endpoints used:

* ``GET {base_url}/cardinfo.php?name=A|B|C`` – exact-name lookup (pipe separated).
* ``GET {base_url}/cardinfo.php``            – the complete catalog (~13k cards).
* ``GET https://images.ygoprodeck.com/...``   – artwork images.

Licensing / terms: YGOPRODeck data and images are provided for non-commercial
use under their API terms; both must be reviewed before any commercial
deployment.  TCGplayer / Cardmarket integrations are NOT required by this
project, and Konami web properties are never scraped.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path

import requests

from app.services.providers.base import ArtworkRecord, CardDataProvider, CardRecord, PrintingRecord

logger = logging.getLogger(__name__)

USER_AGENT = "yugioh-lab-card-recognizer/0.1 (research project; card recognition backend)"
DEFAULT_BASE_URL = "https://db.ygoprodeck.com/api/v7"
DEFAULT_NAME_BATCH_SIZE = 20
DEFAULT_MAX_RETRIES = 4
DOWNLOAD_CHUNK_BYTES = 64 * 1024


class YGOProDeckError(RuntimeError):
    """A request to the provider failed permanently."""


class YGOProDeckNotFound(YGOProDeckError):
    """The API answered HTTP 400 ("No card matching your query was found")."""


# ------------------------------------------------------------------ parsing


def _opt_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _opt_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def parse_card(payload: dict) -> CardRecord:
    """Map one raw ``cardinfo.php`` card object to a :class:`CardRecord`.

    ``def`` (a Python keyword) is exposed as ``def_``; ``frameType`` becomes
    ``frame_type`` and ``desc`` becomes ``description``.
    """
    artworks = [
        ArtworkRecord(
            id=int(image["id"]),
            image_url=_opt_str(image.get("image_url")),
            image_url_small=_opt_str(image.get("image_url_small")),
            image_url_cropped=_opt_str(image.get("image_url_cropped")),
        )
        for image in payload.get("card_images") or []
        if image.get("id") is not None
    ]
    printings = [
        PrintingRecord(
            set_code=str(card_set.get("set_code") or "").strip(),
            set_name=str(card_set.get("set_name") or "").strip(),
            rarity=_opt_str(card_set.get("set_rarity")),
            rarity_code=_opt_str(card_set.get("set_rarity_code")),
        )
        for card_set in payload.get("card_sets") or []
        if card_set.get("set_code")
    ]
    return CardRecord(
        id=int(payload["id"]),
        name=str(payload["name"]).strip(),
        type=_opt_str(payload.get("type")),
        frame_type=_opt_str(payload.get("frameType")),
        description=payload.get("desc"),
        race=_opt_str(payload.get("race")),
        attribute=_opt_str(payload.get("attribute")),
        level=_opt_int(payload.get("level")),
        atk=_opt_int(payload.get("atk")),
        def_=_opt_int(payload.get("def")),
        archetype=_opt_str(payload.get("archetype")),
        artworks=artworks,
        printings=printings,
    )


def parse_response(payload: dict) -> list[CardRecord]:
    """Parse a whole ``{"data": [...]}`` response.  Malformed entries are skipped with a warning."""
    records: list[CardRecord] = []
    for item in payload.get("data") or []:
        try:
            records.append(parse_card(item))
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Skipping malformed card payload (%s): %r", exc, item.get("id") if isinstance(item, dict) else item)
    return records


# --------------------------------------------------------------- http client


class RetryingHttpClient:
    """Small GET client with a politeness delay and exponential backoff.

    Retries on 429, 5xx, connection errors and timeouts (``max_retries`` times).
    ``sleep`` / ``clock`` are injectable so tests run instantly.
    """

    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        request_delay_seconds: float = 0.15,
        timeout_seconds: float = 60.0,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base_seconds: float = 1.0,
        backoff_max_seconds: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._session = session or requests.Session()
        self._session.headers.setdefault("User-Agent", USER_AGENT)
        self._delay = max(0.0, request_delay_seconds)
        self._timeout = timeout_seconds
        self._max_retries = max(0, max_retries)
        self._backoff_base = backoff_base_seconds
        self._backoff_max = backoff_max_seconds
        self._sleep = sleep
        self._clock = clock
        self._last_request_at: float | None = None

    def _throttle(self) -> None:
        if self._last_request_at is not None and self._delay > 0:
            remaining = self._delay - (self._clock() - self._last_request_at)
            if remaining > 0:
                self._sleep(remaining)
        self._last_request_at = self._clock()

    def _backoff(self, attempt: int, response: requests.Response | None = None) -> None:
        wait = min(self._backoff_base * (2**attempt), self._backoff_max)
        retry_after = response.headers.get("Retry-After") if response is not None else None
        if retry_after and retry_after.isdigit():
            wait = min(max(wait, float(retry_after)), self._backoff_max)
        logger.warning("Retrying in %.1fs (attempt %d/%d)", wait, attempt + 1, self._max_retries)
        self._sleep(wait)

    def get(self, url: str, *, params: dict | None = None, stream: bool = False) -> requests.Response:
        """GET ``url``; raises :class:`YGOProDeckNotFound` on 400 and :class:`YGOProDeckError` otherwise."""
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            self._throttle()
            try:
                response = self._session.get(url, params=params, timeout=self._timeout, stream=stream)
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_error = exc
                logger.warning("Request to %s failed: %s", url, exc)
                if attempt < self._max_retries:
                    self._backoff(attempt)
                continue
            status = response.status_code
            if status == 429 or 500 <= status < 600:
                last_error = YGOProDeckError(f"HTTP {status} from {url}")
                response.close()
                if attempt < self._max_retries:
                    self._backoff(attempt, response)
                continue
            if status == 400:
                response.close()
                raise YGOProDeckNotFound(f"HTTP 400 from {url}: {_error_message(response)}")
            if status >= 400:
                response.close()
                raise YGOProDeckError(f"HTTP {status} from {url}")
            return response
        raise YGOProDeckError(f"Giving up on {url} after {self._max_retries + 1} attempts") from last_error


def _error_message(response: requests.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:200] if isinstance(response.text, str) else ""
    if isinstance(body, dict):
        return str(body.get("error") or body)
    return str(body)


def download_to_path(
    client: RetryingHttpClient, url: str, destination: Path, *, overwrite: bool = False
) -> Path:
    """Stream ``url`` into ``destination`` (written via a temporary ``.part`` file)."""
    if destination.exists() and not overwrite:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    response = client.get(url, stream=True)
    try:
        with partial.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_BYTES):
                if chunk:
                    handle.write(chunk)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    finally:
        response.close()
    partial.replace(destination)
    return destination


# ------------------------------------------------------------------ provider


def chunked(items: Sequence[str], size: int) -> Iterator[list[str]]:
    """Yield consecutive chunks of at most ``size`` items."""
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


class YGOProDeckProvider(CardDataProvider):
    """Card metadata + artwork downloads from the public YGOPRODeck API."""

    name = "ygoprodeck"

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        request_delay_seconds: float = 0.15,
        timeout_seconds: float = 60.0,
        session: requests.Session | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        name_batch_size: int = DEFAULT_NAME_BATCH_SIZE,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = RetryingHttpClient(
            session=session,
            request_delay_seconds=request_delay_seconds,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            sleep=sleep,
        )
        self._name_batch_size = max(1, name_batch_size)
        #: Names requested by the last :meth:`fetch_cards_by_names` call that the API did not know.
        self.missing_names: list[str] = []

    @property
    def cardinfo_url(self) -> str:
        return f"{self.base_url}/cardinfo.php"

    def _fetch_names(self, names: list[str]) -> list[CardRecord]:
        response = self._client.get(self.cardinfo_url, params={"name": "|".join(names)})
        return parse_response(response.json())

    def fetch_cards_by_names(self, names: Iterable[str]) -> list[CardRecord]:
        """Fetch cards by exact name in pipe-joined batches.

        A batch answered with HTTP 400 (one unknown name poisons the whole batch)
        is retried name by name; unknown names are logged, recorded in
        :attr:`missing_names` and skipped.
        """
        unique = list(dict.fromkeys(name.strip() for name in names if name and name.strip()))
        self.missing_names = []
        records: list[CardRecord] = []
        for batch in chunked(unique, self._name_batch_size):
            try:
                records.extend(self._fetch_names(batch))
                continue
            except YGOProDeckNotFound:
                logger.info("Batch of %d names contained unknown cards; retrying individually", len(batch))
            for name in batch:
                try:
                    records.extend(self._fetch_names([name]))
                except YGOProDeckNotFound as exc:
                    logger.warning("Card not found on YGOPRODeck: %r (%s)", name, exc)
                    self.missing_names.append(name)
        return records

    def iter_all_cards(self) -> Iterator[CardRecord]:
        """Stream the complete catalog (a single large JSON response)."""
        response = self._client.get(self.cardinfo_url, stream=True)
        try:
            payload = response.json()
        finally:
            response.close()
        items = payload.get("data") or []
        logger.info("YGOPRODeck catalog: %d cards", len(items))
        for item in items:
            try:
                yield parse_card(item)
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Skipping malformed catalog entry (%s): %r", exc, item.get("id") if isinstance(item, dict) else item)

    def download_artwork(self, artwork: ArtworkRecord, destination: Path, *, overwrite: bool = False) -> Path | None:
        """Download the full-size artwork unless ``destination`` already exists."""
        if destination.exists() and not overwrite:
            return destination
        if not artwork.image_url:
            logger.warning("Artwork %s has no image_url; nothing to download", artwork.id)
            return None
        return download_to_path(self._client, artwork.image_url, destination, overwrite=overwrite)
