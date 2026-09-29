from pathlib import Path

import cv2


# We'll test against one of our downloaded Blue-Eyes images.
INPUT_IMAGE = Path("data/cards/89631139.jpg")

OUTPUT_DIR = Path("data/debug")

# Standard size we'll use for every normalized Yu-Gi-Oh card.
CARD_WIDTH = 421
CARD_HEIGHT = 614


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(INPUT_IMAGE))

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {INPUT_IMAGE}"
        )

    # --------------------------------------------------
    # Normalize every card to the same dimensions.
    # --------------------------------------------------

    card = cv2.resize(
        image,
        (CARD_WIDTH, CARD_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )

    # --------------------------------------------------
    # Initial regions of interest (ROI)
    #
    # These coordinates are intentionally approximate.
    # We'll look at the output and adjust them.
    # --------------------------------------------------

    # Card name near the top.
    name_x1 = 20
    name_y1 = 25
    name_x2 = 350
    name_y2 = 80
    # Set code near the right side below the artwork.
    set_x1 = 245
    set_y1 = 450
    set_x2 = 415
    set_y2 = 495

    # Crop the regions.
    name_region = card[
        name_y1:name_y2,
        name_x1:name_x2,
    ]

    set_code_region = card[
        set_y1:set_y2,
        set_x1:set_x2,
    ]

    # --------------------------------------------------
    # Save the individual crops.
    # --------------------------------------------------

    cv2.imwrite(
        str(OUTPUT_DIR / "name_region.jpg"),
        name_region,
    )

    cv2.imwrite(
        str(OUTPUT_DIR / "set_code_region.jpg"),
        set_code_region,
    )

    # --------------------------------------------------
    # Create a debug image showing where we cropped.
    # --------------------------------------------------

    debug = card.copy()

    cv2.rectangle(
        debug,
        (name_x1, name_y1),
        (name_x2, name_y2),
        (0, 255, 0),
        2,
    )

    cv2.rectangle(
        debug,
        (set_x1, set_y1),
        (set_x2, set_y2),
        (255, 0, 0),
        2,
    )

    cv2.imwrite(
        str(OUTPUT_DIR / "debug_regions.jpg"),
        debug,
    )

    cv2.imwrite(
        str(OUTPUT_DIR / "normalized_card.jpg"),
        card,
    )

    print("Finished.")
    print()
    print("Created:")
    print("  data/debug/normalized_card.jpg")
    print("  data/debug/debug_regions.jpg")
    print("  data/debug/name_region.jpg")
    print("  data/debug/set_code_region.jpg")


if __name__ == "__main__":
    main()