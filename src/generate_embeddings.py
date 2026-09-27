import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


MODEL_NAME = "openai/clip-vit-base-patch32"

CARDS_FILE = Path("data/cards.json")
OUTPUT_FILE = Path("data/embeddings.npz")


def main():
    print("Loading CLIP model...")

    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME)

    model.eval()

    cards = json.loads(
        CARDS_FILE.read_text(encoding="utf-8")
    )

    embeddings = []
    names = []
    ids = []

    for card in cards:
        print(f"Embedding: {card['name']}")

        image = Image.open(card["image"]).convert("RGB")

        inputs = processor(
            images=image,
            return_tensors="pt",
        )

        with torch.inference_mode():
            embedding = model.get_image_features(**inputs)

        # Normalize the vector.
        embedding = embedding / embedding.norm(
            p=2,
            dim=-1,
            keepdim=True,
        )

        embeddings.append(
            embedding.squeeze(0).cpu().numpy()
        )

        names.append(card["name"])
        ids.append(card["id"])

    embeddings = np.stack(embeddings)

    np.savez(
        OUTPUT_FILE,
        embeddings=embeddings,
        names=np.array(names),
        ids=np.array(ids),
    )

    print()
    print("Finished!")
    print(f"Embedding matrix shape: {embeddings.shape}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()