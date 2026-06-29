"""
Project Configuration File
--------------------------
This file stores all configurable parameters used throughout the project.

Changing a value here automatically updates the behavior of the entire application.
"""

from pathlib import Path

# ==========================================================
# Project Directories
# ==========================================================

# Root project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Folder containing sample thermal images
DATA_DIR = PROJECT_ROOT / "data" / "sample_images"

# Folder where processed images will be saved
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ==========================================================
# Image Processing Parameters
# ==========================================================

# Percentile used for hotspot detection
HOTSPOT_PERCENTILE = 99.85

# Ignore very small contours
MIN_HOTSPOT_AREA = 40

# Kernel used for erosion
EROSION_KERNEL_SIZE = (30, 30)

# Kernel used for morphology operations
MORPH_KERNEL_SIZE = (9, 9)

# ==========================================================
# Solar Panel Polygon Coordinates
# ==========================================================

PANEL_POLYGON = [
    (115, 35),
    (260, 15),
    (360, 405),
    (15, 460)
]

# ==========================================================
# Exclusion Region
# ==========================================================

EXCLUSION_POLYGON = [
    (22, 329),
    (332, 269),
    (332, 277),
    (22, 337)
]