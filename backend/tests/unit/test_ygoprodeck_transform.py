"""YGOPRODeck payload parsing + provider behaviour (network fully mocked)."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
import requests

from app.services.providers.base import CardRecord
from app.services.providers.ygoprodeck import (
    YGOProDeckError,
    YGOProDeckProvider,
    parse_card,
    parse_response,
)

ARMADES_ID = 88033975


# ------------------------------------------------------------------ parsing


def _by_name(records: list[CardRecord], name: str) -> CardRecord:
    return next(record for record in records if record.name == name)


def test_parse_response_returns_all_fixture_cards(ygoprodeck_sample: dict) -> None:
    records = parse_response(ygoprodeck_sample)
    assert len(records) == 7
    assert {record.id for record in records} == {card["id"] for card in ygoprodeck_sample["data"]}


def test_parse_card_armades_fields_and_printings(ygoprodeck_sample: dict) -> None:
    payload = next(card for card in ygoprodeck_sample["data"] if card["id"] == ARMADES_ID)
    record = parse_card(payload)

    assert record.id == ARMADES_ID
    assert record.name == "Armades, Keeper of Boundaries"
    assert record.type == "Synchro Monster"
    assert record.frame_type == "synchro"
    assert record.description.startswith("1 Tuner + 1 or more non-Tuner monsters")
    assert record.race == "Fiend"
    assert record.attribute == "LIGHT"
    assert (record.level, record.atk, record.def_) == (5, 2300, 1500)
    assert record.archetype is None

    assert len(record.printings) == 5
    jotl = next(p for p in record.printings if p.set_code == "JOTL-EN045")
    assert jotl.set_name == "Judgment of the Light"
    assert jotl.rarity == "Secret Rare"
    assert jotl.rarity_code == "ScR"  # provider value "(ScR)" is stored without parentheses
    # BLGG-EN090 exists in two rarities and both must survive parsing.
    blgg = [p for p in record.printings if p.set_code == "BLGG-EN090"]
    assert {p.rarity for p in blgg} == {"Secret Rare", "Starlight Rare"}

    assert len(record.artworks) == 1
    assert record.artworks[0].id == ARMADES_ID
    assert record.artworks[0].image_url == "https://images.ygoprodeck.com/images/cards/88033975.jpg"
    assert record.artworks[0].image_url_small.endswith("cards_small/88033975.jpg")
    assert record.artworks[0].image_url_cropped.endswith("cards_cropped/88033975.jpg")


def test_parse_card_spell_without_monster_fields(ygoprodeck_sample: dict) -> None:
    record = _by_name(parse_response(ygoprodeck_sample), "Pot of Greed")
    assert record.type == "Spell Card"
    assert record.attribute is None
    assert record.level is None and record.atk is None and record.def_ is None
    # empty rarity codes become None instead of ""
    qcsr = next(p for p in record.printings if p.set_code == "TBC1-ENS01")
    assert qcsr.rarity == "Quarter Century Secret Rare"
    assert qcsr.rarity_code is None


def test_parse_card_def_keyword_maps_to_def_() -> None:
    record = parse_card({"id": 1, "name": "X", "def": "200", "atk": 100, "level": "4"})
    assert record.def_ == 200
    assert record.atk == 100
    assert record.level == 4
    assert record.printings == [] and record.artworks == []


def test_parse_response_skips_malformed_entries() -> None:
    records = parse_response({"data": [{"id": 1, "name": "ok"}, {"name": "missing id"}, {"id": "abc", "name": "bad"}]})
    assert [record.id for record in records] == [1]


# ------------------------------------------------------------ fake network


class FakeResponse:
    """Minimal stand-in for :class:`requests.Response`."""

    def __init__(self, status_code: int, payload: object | None = None, *, content: bytes = b"", headers=None):
        self.status_code = status_code
        self._payload = payload
        self._content = content
        self.headers = headers or {}
        self.closed = False

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload

    @property
    def text(self) -> str:
        return json.dumps(self._payload) if self._payload is not None else ""

    def iter_content(self, chunk_size: int = 1):
        for start in range(0, len(self._content), chunk_size):
            yield self._content[start : start + chunk_size]

    def close(self) -> None:
        self.closed = True


def _make_card(card_id: int, name: str) -> dict:
    return {"id": card_id, "name": name, "type": "Spell Card", "card_sets": [], "card_images": []}


def _install_fake_get(monkeypatch: pytest.MonkeyPatch, handler):
    """Patch ``requests.Session.get`` with ``handler(url, params)`` and record the calls."""
    calls: list[tuple[str, dict | None, dict]] = []

    def fake_get(self, url, params=None, **kwargs):
        calls.append((url, params, kwargs))
        return handler(url, params)

    monkeypatch.setattr(requests.Session, "get", fake_get)
    return calls


def _provider(**kwargs) -> YGOProDeckProvider:
    return YGOProDeckProvider("https://example.test/api/v7/", request_delay_seconds=0, sleep=lambda _s: None, **kwargs)


def test_fetch_cards_by_names_uses_pipe_joined_query(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(url, params):
        names = params["name"].split("|")
        return FakeResponse(200, {"data": [_make_card(i + 1, n) for i, n in enumerate(names)]})

    calls = _install_fake_get(monkeypatch, handler)
    provider = _provider()
    records = provider.fetch_cards_by_names(["Pot of Greed", "Mirror Force", "Pot of Greed", " Kuriboh "])

    assert [record.name for record in records] == ["Pot of Greed", "Mirror Force", "Kuriboh"]
    assert len(calls) == 1
    url, params, kwargs = calls[0]
    assert url == "https://example.test/api/v7/cardinfo.php"
    assert params == {"name": "Pot of Greed|Mirror Force|Kuriboh"}
    assert kwargs["timeout"] == 60.0
    assert provider.missing_names == []


def test_fetch_cards_by_names_batches_in_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake_get(monkeypatch, lambda url, params: FakeResponse(200, {"data": []}))
    provider = _provider(name_batch_size=3)
    provider.fetch_cards_by_names([f"Card {i}" for i in range(7)])
    assert [len(params["name"].split("|")) for _, params, _ in calls] == [3, 3, 1]


def test_fetch_cards_by_names_falls_back_per_name_on_400(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(url, params):
        names = params["name"].split("|")
        if any(name == "Nonexistent Card" for name in names):
            return FakeResponse(400, {"error": "No card matching your query was found in the database."})
        return FakeResponse(200, {"data": [_make_card(hash(n) % 1000, n) for n in names]})

    calls = _install_fake_get(monkeypatch, handler)
    provider = _provider()
    records = provider.fetch_cards_by_names(["Pot of Greed", "Nonexistent Card", "Kuriboh"])

    assert [record.name for record in records] == ["Pot of Greed", "Kuriboh"]
    assert provider.missing_names == ["Nonexistent Card"]
    # one batched call + three per-name retries
    assert [params["name"] for _, params, _ in calls] == [
        "Pot of Greed|Nonexistent Card|Kuriboh",
        "Pot of Greed",
        "Nonexistent Card",
        "Kuriboh",
    ]


def test_retries_on_server_errors_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter([FakeResponse(503), FakeResponse(429, headers={"Retry-After": "2"}), FakeResponse(200, {"data": [_make_card(1, "A")]})])
    calls = _install_fake_get(monkeypatch, lambda url, params: next(responses))
    sleeps: list[float] = []
    provider = YGOProDeckProvider("https://example.test/api/v7", request_delay_seconds=0, sleep=sleeps.append)
    records = provider.fetch_cards_by_names(["A"])
    assert [record.name for record in records] == ["A"]
    assert len(calls) == 3
    assert len(sleeps) == 2 and sleeps[1] >= 2.0  # Retry-After honoured


def test_gives_up_after_max_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake_get(monkeypatch, lambda url, params: FakeResponse(500))
    provider = _provider(max_retries=2)
    with pytest.raises(YGOProDeckError):
        provider.fetch_cards_by_names(["A"])
    assert len(calls) == 3


def test_retries_on_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {"n": 0}

    def handler(url, params):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise requests.ConnectionError("boom")
        return FakeResponse(200, {"data": []})

    _install_fake_get(monkeypatch, handler)
    assert _provider().fetch_cards_by_names(["A"]) == []
    assert attempts["n"] == 2


def test_iter_all_cards_streams_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake_get(monkeypatch, lambda url, params: FakeResponse(200, {"data": [_make_card(1, "A"), _make_card(2, "B")]}))
    names = [record.name for record in _provider().iter_all_cards()]
    assert names == ["A", "B"]
    assert calls[0][1] is None and calls[0][2]["stream"] is True


def test_download_artwork_streams_to_file_and_skips_existing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payload = b"\xff\xd8" + b"x" * 5000
    calls = _install_fake_get(monkeypatch, lambda url, params: FakeResponse(200, content=payload))
    provider = _provider()
    record = parse_card({"id": 7, "name": "Seven", "card_images": [{"id": 7, "image_url": "https://images.test/7.jpg"}]})
    destination = tmp_path / "7.jpg"

    assert provider.download_artwork(record.artworks[0], destination) == destination
    assert destination.read_bytes() == payload
    assert not destination.with_name("7.jpg.part").exists()
    assert calls[0][0] == "https://images.test/7.jpg" and calls[0][2]["stream"] is True

    # second call: file exists -> no request
    assert provider.download_artwork(record.artworks[0], destination) == destination
    assert len(calls) == 1


def test_download_artwork_without_url_returns_none(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = _install_fake_get(monkeypatch, lambda url, params: FakeResponse(200))
    record = parse_card({"id": 8, "name": "Eight", "card_images": [{"id": 8}]})
    assert _provider().download_artwork(record.artworks[0], tmp_path / "8.jpg") is None
    assert calls == []


def test_user_agent_header_is_set() -> None:
    session = requests.Session()
    YGOProDeckProvider(session=session)
    assert "yugioh-lab" in session.headers["User-Agent"]


def test_query_is_url_encoded_by_requests() -> None:
    # Sanity check of the assumption documented in the provider: requests encodes the params.
    prepared = requests.Request("GET", "https://example.test/cardinfo.php", params={"name": "Armades, Keeper of Boundaries|Pot of Greed"}).prepare()
    query = parse_qs(urlsplit(prepared.url).query)
    assert query["name"] == ["Armades, Keeper of Boundaries|Pot of Greed"]
