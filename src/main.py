"""
main.py
--------
Entry point of the Solar Panel Hotspot Detection project.
"""

from pathlib import Path
import matplotlib.pyplot as plt

from config import DATA_DIR
from image_loader import (
    load_image,
    normalize_image,
    prepare_display_image,
)


def main():
    """
    Main function.
    """

    # Search for all TIFF images
    image_paths = list(DATA_DIR.glob("*.tif"))
    image_paths.extend(DATA_DIR.glob("*.tiff"))

    if not image_paths:
        raise FileNotFoundError(
            "No TIFF images found inside data/sample_images/"
        )

    # Select the first image
    image_path = image_paths[0]

    print(f"Loading: {image_path.name}")

    # Load image
    image = load_image(image_path)

    # Normalize image
    normalized_image = normalize_image(image)

    # Prepare for display
    display_image = prepare_display_image(normalized_image)

    # Display image
    plt.figure(figsize=(8, 10))
    plt.imshow(display_image)
    plt.title("Loaded Thermal Image")
    plt.axis("on")
    plt.show()


if __name__ == "__main__":
    main()