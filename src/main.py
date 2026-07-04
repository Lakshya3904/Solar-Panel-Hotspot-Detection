"""
main.py
--------
Entry point of the Solar Panel Hotspot Detection project.
"""

import cv2
import matplotlib.pyplot as plt

from config import (
    DATA_DIR,
    PANEL_POLYGON,
    EROSION_KERNEL_SIZE,
    EXCLUSION_POLYGON,
    HOTSPOT_PERCENTILE
)

from image_loader import (
    load_image,
    normalize_image,
    prepare_display_image
)

from masks import (
    create_panel_mask,
    draw_panel_boundary,
    create_inner_mask,
    remove_separator_lines,
    apply_exclusion_region
)

from hotspot_detector import (
    calculate_hotspot_threshold,
    generate_hotspot_mask
)


def main():

    image_paths = list(DATA_DIR.glob("*.tif"))
    image_paths.extend(DATA_DIR.glob("*.tiff"))

    if not image_paths:
        raise FileNotFoundError(
            "No TIFF images found."
        )

    image_path = image_paths[0]

    print(f"Loading {image_path.name}")

    image = load_image(image_path)

    normalized_image = normalize_image(image)

    display_image = prepare_display_image(
        normalized_image
    )

    gray_image = cv2.cvtColor(
        display_image,
        cv2.COLOR_RGB2GRAY
    )

    panel_mask = create_panel_mask(
        gray_image.shape,
        PANEL_POLYGON
    )

    inner_mask = create_inner_mask(
        panel_mask,
        EROSION_KERNEL_SIZE
    )

    roi_mask = remove_separator_lines(
        inner_mask
    )

    roi_mask = apply_exclusion_region(
        roi_mask,
        EXCLUSION_POLYGON
    )

    threshold = calculate_hotspot_threshold(
    gray_image,
    roi_mask,
    HOTSPOT_PERCENTILE
    )

    hotspot_mask = generate_hotspot_mask(
    gray_image,
    roi_mask,
    threshold
    )
    output = draw_panel_boundary(
        display_image,
        PANEL_POLYGON
    )

    plt.figure(figsize=(22,6))

    plt.subplot(1,4,1)
    plt.imshow(panel_mask,cmap="gray")
    plt.title("Panel Mask")
    plt.axis("off")

    plt.subplot(1,4,2)
    plt.imshow(roi_mask,cmap="gray")
    plt.title("ROI Mask")
    plt.axis("off")

    plt.subplot(1,4,3)
    plt.imshow(hotspot_mask,cmap="gray")
    plt.title("Hotspot Mask")
    plt.axis("off")

    plt.subplot(1,4,4)
    plt.imshow(output)
    plt.title("Panel Boundary")
    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    main()