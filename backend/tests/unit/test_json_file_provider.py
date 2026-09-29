"""JsonFileProvider on both supported JSON layouts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.providers.base import ArtworkRecord
from app.services.providers.json_file import JsonFileProvider, is_api_shaped, parse_legacy_card

LEGACY_CARDS = [
    {
        "id": 88033975,
        "name": "Armades, Keeper of Boundaries",
        "type": "Synchro Monster",
        "frame_type": "synchro",
        "description": "1 Tuner + 1 or more non-Tuner monsters",
        "artworks": [
            {
                "id": 88033975,
                "image": "data\\cards\\88033975.jpg",
                "image_url": "https://images.ygoprodeck.com/images/cards/88033975.jpg",
            }
        ],
        "printings": [
            {"set_name": "Judgment of the Light", "set_code": "JOTL-EN045", "rarity": "Secret Rare", "rarity_code": "(ScR)"},
            {"set_name": "Battles of Legend: Glorious Gallery", "set_code": "BLGG-EN090", "rarity": "Starlight Rare", "rarity_code": ""},
        ],
    },
    {
        "id": 55144522,
        "name": "Pot of Greed",
        "type": "Spell Card",
        "frame_type": "spell",
        "description": "Draw 2 cards.",
        "artworks": [{"id": 55144522, "image": "data\\cards\\missing.jpg", "image_url": "https://images.test/55144522.jpg"}],
        "printings": [],
    },
]


@pytest.fixture()
def legacy_file(tmp_path: Path) -> Path:
    """A legacy catalog laid out like the repository: <root>/data/cards.json + <root>/data/cards/*.jpg."""
    data_dir = tmp_path / "data"
    (data_dir / "cards").mkdir(parents=True)
    (data_dir / "cards" / "88033975.jpg").write_bytes(b"legacy-jpeg")
    path = data_dir / "cards.json"
    path.write_text(json.dumps(LEGACY_CARDS), encoding="utf-8")
    return path


def test_is_api_shaped_detection() -> None:
    assert is_api_shaped({"id": 1, "name": "x", "card_sets": []})
    assert is_api_shaped({"id": 1, "name": "x", "frameType": "spell"})
    assert not is_api_shaped(LEGACY_CARDS[0])


def test_parse_legacy_card() -> None:
    record, local_images = parse_legacy_card(LEGACY_CARDS[0])
    assert record.id == 88033975
    assert record.frame_type == "synchro"
    assert record.description == "1 Tuner + 1 or more non-Tuner monsters"
    assert record.race is None and record.def_ is None
    assert [(p.set_code, p.set_name, p.rarity, p.rarity_code) for p in record.printings] == [
        ("JOTL-EN045", "Judgment of the Light", "Secret Rare", "ScR"),
        ("BLGG-EN090", "Battles of Legend: Glorious Gallery", "Starlight Rare", None),
    ]
    assert record.artworks == [ArtworkRecord(88033975, "https://images.ygoprodeck.com/images/cards/88033975.jpg")]
    assert local_images == {88033975: "data\\cards\\88033975.jpg"}


def test_legacy_format_end_to_end(legacy_file: Path) -> None:
    provider = JsonFileProvider(legacy_file)
    names = [record.name for record in provider.iter_all_cards()]
    assert names == ["Armades, Keeper of Boundaries", "Pot of Greed"]

    found = provider.fetch_cards_by_names(["pot of greed", "Nope"])
    assert [record.name for record in found] == ["Pot of Greed"]

    # the Windows-style relative image path resolves relative to the JSON file's parent directory
    assert provider.local_image_path(88033975) == (legacy_file.parent / "cards" / "88033975.jpg").resolve()
    assert provider.local_image_path(55144522) is None


def test_api_dump_format(fixtures_dir: Path) -> None:
    provider = JsonFileProvider(fixtures_dir / "ygoprodeck_sample.json")
    records = list(provider.iter_all_cards())
    assert len(records) == 7
    armades = provider.fetch_cards_by_names(["Armades, Keeper of Boundaries"])
    assert len(armades) == 1
    assert len(armades[0].printings) == 5
    assert armades[0].def_ == 1500
    assert provider.local_image_path(88033975) is None


def test_bare_list_of_api_cards(tmp_path: Path) -> None:
    path = tmp_path / "bare.json"
    path.write_text(json.dumps([{"id": 1, "name": "A", "frameType": "spell", "card_sets": [], "card_images": []}]), encoding="utf-8")
    records = list(JsonFileProvider(path).iter_all_cards())
    assert [(r.id, r.frame_type) for r in records] == [(1, "spell")]


def test_download_artwork_copies_local_file(legacy_file: Path, tmp_path: Path) -> None:
    provider = JsonFileProvider(legacy_file)
    record = provider.fetch_cards_by_names(["Armades, Keeper of Boundaries"])[0]
    destination = tmp_path / "cache" / "88033975.jpg"

    assert provider.download_artwork(record.artworks[0], destination) == destination
    assert destination.read_bytes() == b"legacy-jpeg"

    # destination == source: nothing to copy, still reported as present
    source = provider.local_image_path(88033975)
    assert provider.download_artwork(record.artworks[0], source) == source


def test_download_artwork_falls_back_to_http(legacy_file: Path, tmp_path: Path) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.urls: list[str] = []

        def get(self, url: str, *, params=None, stream: bool = False):
            self.urls.append(url)

            class Response:
                def iter_content(self, chunk_size: int):
                    yield b"downloaded"

                def close(self) -> None:
                    pass

            return Response()

    client = FakeClient()
    provider = JsonFileProvider(legacy_file, http_client_factory=lambda: client)
    record = provider.fetch_cards_by_names(["Pot of Greed"])[0]
    destination = tmp_path / "55144522.jpg"
    assert provider.download_artwork(record.artworks[0], destination) == destination
    assert destination.read_bytes() == b"downloaded"
    assert client.urls == ["https://images.test/55144522.jpg"]

    # existing destination short-circuits without any request
    assert provider.download_artwork(record.artworks[0], destination) == destination
    assert len(client.urls) == 1


def test_invalid_layout_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"cards": []}), encoding="utf-8")
    provider = JsonFileProvider(path)
    assert list(provider.iter_all_cards()) == []  # {"cards": ...} has no "data" -> empty catalog
    path.write_text(json.dumps("nonsense"), encoding="utf-8")
    with pytest.raises(ValueError):
        list(JsonFileProvider(path).iter_all_cards())
