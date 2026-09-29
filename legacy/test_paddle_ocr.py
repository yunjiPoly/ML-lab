from pathlib import Path

from paddleocr import TextRecognition


NAME_IMAGE = Path("data/debug/photo_name.jpg")
SET_CODE_IMAGE = Path("data/debug/photo_set_code.jpg")


def recognize(model, label, image_path):
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    print()
    print("=" * 60)
    print(label)
    print(f"Image: {image_path}")
    print("=" * 60)

    results = model.predict(
        input=str(image_path),
        batch_size=1,
    )

    for result in results:
        result.print()


def main():
    print("Loading PP-OCRv6...")
    print("The first run may download the recognition model.")

    model = TextRecognition(
        model_name="PP-OCRv6_medium_rec",
        device="cpu",
    )

    recognize(
        model,
        "CARD NAME",
        NAME_IMAGE,
    )

    recognize(
        model,
        "SET CODE",
        SET_CODE_IMAGE,
    )


if __name__ == "__main__":
    main()