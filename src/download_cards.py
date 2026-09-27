import json
import time
from pathlib import Path

import requests


API_URL = "https://db.ygoprodeck.com/api/v7/cardinfo.php"

DATA_DIR = Path("data")
CARD_DIR = DATA_DIR / "cards"

CARD_NAMES = [
    "Blue-Eyes White Dragon",
    "Dark Magician",
    "Dark Magician Girl",
    "Red-Eyes Black Dragon",
    "Exodia the Forbidden One",
    "Kuriboh",
    "Monster Reborn",
    "Pot of Greed",
    "Mirror Force",
    "Polymerization",
]


def download_card(name: str):
    print(f"Downloading: {name}")

    response = requests.get(
        API_URL,
        params={"name": name},
        timeout=30,
    )
    response.raise_for_status()

    card = response.json()["data"][0]

    card_image = card["card_images"][0]

    image_id = card_image["id"]
    image_url = card_image["image_url"]

    image_response = requests.get(
        image_url,
        timeout=30,
    )
    image_response.raise_for_status()

    filename = CARD_DIR / f"{image_id}.jpg"
    filename.write_bytes(image_response.content)

    return {
        "id": image_id,
        "name": card["name"],
        "type": card["type"],
        "image": str(filename),
    }


def main():
    CARD_DIR.mkdir(parents=True, exist_ok=True)

    cards = []

    for name in CARD_NAMES:
        card = download_card(name)
        cards.append(card)

        time.sleep(0.1)

    catalog_file = DATA_DIR / "cards.json"

    catalog_file.write_text(
        json.dumps(cards, indent=2),
        encoding="utf-8",
    )

    print()
    print(f"Downloaded {len(cards)} cards.")
    print(f"Catalog saved to: {catalog_file}")


if __name__ == "__main__":
    main()