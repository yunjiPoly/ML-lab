from pathlib import Path

import cv2
import numpy as np


INPUT_IMAGE = Path("data/test/172026.jpg")
OUTPUT_DIR = Path("data/debug")

CARD_WIDTH = 421
CARD_HEIGHT = 614


def order_points(points):
    """
    Return corners in this order:

    top-left
    top-right
    bottom-right
    bottom-left
    """

    points = np.array(points, dtype=np.float32)

    rect = np.zeros((4, 2), dtype=np.float32)

    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)

    rect[0] = points[np.argmin(sums)]
    rect[2] = points[np.argmax(sums)]

    rect[1] = points[np.argmin(differences)]
    rect[3] = points[np.argmax(differences)]

    return rect


def find_card(image):
    """
    Find a large rectangular contour that looks
    approximately like a Yu-Gi-Oh card.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    edges = cv2.Canny(
        blurred,
        50,
        150,
    )

    edges = cv2.dilate(
        edges,
        np.ones((3, 3), np.uint8),
        iterations=1,
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    # Largest contours first.
    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True,
    )

    image_area = image.shape[0] * image.shape[1]

    for contour in contours:
        area = cv2.contourArea(contour)

        # Ignore small objects.
        if area < image_area * 0.10:
            continue

        perimeter = cv2.arcLength(
            contour,
            True,
        )

        approx = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True,
        )

        if len(approx) != 4:
            continue

        points = approx.reshape(4, 2)

        ordered = order_points(points)

        top_left, top_right, bottom_right, bottom_left = ordered

        width_top = np.linalg.norm(top_right - top_left)
        width_bottom = np.linalg.norm(bottom_right - bottom_left)

        height_left = np.linalg.norm(bottom_left - top_left)
        height_right = np.linalg.norm(bottom_right - top_right)

        width = (width_top + width_bottom) / 2
        height = (height_left + height_right) / 2

        if height == 0:
            continue

        ratio = width / height

        # Yu-Gi-Oh cards are roughly 0.68 width / height.
        if 0.55 <= ratio <= 0.80:
            return ordered

    return None


def warp_card(image, corners):
    destination = np.array(
        [
            [0, 0],
            [CARD_WIDTH - 1, 0],
            [CARD_WIDTH - 1, CARD_HEIGHT - 1],
            [0, CARD_HEIGHT - 1],
        ],
        dtype=np.float32,
    )

    transform = cv2.getPerspectiveTransform(
        corners,
        destination,
    )

    return cv2.warpPerspective(
        image,
        transform,
        (CARD_WIDTH, CARD_HEIGHT),
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    image = cv2.imread(
        str(INPUT_IMAGE)
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not load {INPUT_IMAGE}"
        )

    print("Searching for card...")

    corners = find_card(image)

    if corners is None:
        print("Could not detect card.")
        return

    print("Card detected.")

    # --------------------------------------------------
    # Debug image showing detected card boundary
    # --------------------------------------------------

    detected = image.copy()

    polygon = corners.astype(np.int32)

    cv2.polylines(
        detected,
        [polygon],
        True,
        (0, 255, 0),
        5,
    )

    cv2.imwrite(
        str(OUTPUT_DIR / "detected_card.jpg"),
        detected,
    )

    # --------------------------------------------------
    # Perspective correction
    # --------------------------------------------------

    card = warp_card(
        image,
        corners,
    )

    cv2.imwrite(
        str(OUTPUT_DIR / "normalized_photo.jpg"),
        card,
    )

    # --------------------------------------------------
    # Regions found from our real-card test
    # --------------------------------------------------

    name_region = card[
        10:65,
        15:360,
    ]

    set_code_region = card[
        435:465,
        290:405,
    ]

    cv2.imwrite(
        str(OUTPUT_DIR / "photo_name.jpg"),
        name_region,
    )

    cv2.imwrite(
        str(OUTPUT_DIR / "photo_set_code.jpg"),
        set_code_region,
    )

    # --------------------------------------------------
    # Debug rectangles
    # --------------------------------------------------

    debug = card.copy()

    cv2.rectangle(
        debug,
        (15, 10),
        (360, 65),
        (0, 255, 0),
        2,
    )

    cv2.rectangle(
        debug,
        (290, 435),
        (405, 465),
        (255, 0, 0),
        2,
    )

    cv2.imwrite(
        str(OUTPUT_DIR / "photo_regions.jpg"),
        debug,
    )

    print()
    print("Created:")
    print("  data/debug/detected_card.jpg")
    print("  data/debug/normalized_photo.jpg")
    print("  data/debug/photo_regions.jpg")
    print("  data/debug/photo_name.jpg")
    print("  data/debug/photo_set_code.jpg")


if __name__ == "__main__":
    main()