import json
import time
from pathlib import Path

import requests


API_URL = "https://db.ygoprodeck.com/api/v7/cardinfo.php"

DATA_DIR = Path("data")
IMAGE_DIR = DATA_DIR / "cards"

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
    "Armades, Keeper of Boundaries",
]


def download_image(url: str, filename: Path):
    """
    Download an image only if we don't already have it.
    """

    if filename.exists():
        print(f"  Image already exists: {filename.name}")
        return

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    filename.write_bytes(response.content)

    print(f"  Saved image: {filename.name}")


def download_card(name: str):
    print()
    print(f"Downloading card data: {name}")

    response = requests.get(
        API_URL,
        params={"name": name},
        timeout=30,
    )

    response.raise_for_status()

    card = response.json()["data"][0]

    # ------------------------------------------------
    # ARTWORKS
    # ------------------------------------------------

    artworks = []

    for image in card.get("card_images", []):
        image_id = image["id"]

        filename = IMAGE_DIR / f"{image_id}.jpg"

        download_image(
            image["image_url"],
            filename,
        )

        artworks.append(
            {
                "id": image_id,
                "image": str(filename),
                "image_url": image["image_url"],
            }
        )

    # ------------------------------------------------
    # PRINTINGS / SETS
    # ------------------------------------------------

    printings = []

    for printing in card.get("card_sets", []):
        printings.append(
            {
                "set_name": printing.get("set_name"),
                "set_code": printing.get("set_code"),
                "rarity": printing.get("set_rarity"),
                "rarity_code": printing.get("set_rarity_code"),
            }
        )

    return {
        "id": card["id"],
        "name": card["name"],
        "type": card["type"],
        "frame_type": card.get("frameType"),
        "description": card.get("desc"),
        "artworks": artworks,
        "printings": printings,
    }


def main():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    cards = []

    for name in CARD_NAMES:
        card = download_card(name)
        cards.append(card)

        print(
            f"  Found {len(card['artworks'])} artwork(s), "
            f"{len(card['printings'])} printing(s)"
        )

        # Stay comfortably below API rate limits.
        time.sleep(0.2)

    catalog_file = DATA_DIR / "cards.json"

    catalog_file.write_text(
        json.dumps(
            cards,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 60)
    print(f"Downloaded {len(cards)} cards.")
    print(f"Catalog saved to: {catalog_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()