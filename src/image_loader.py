"""
image_loader.py
----------------
Functions for loading and preparing thermal images.
"""

import cv2
import tifffile
import numpy as np
from pathlib import Path


def load_image(image_path: Path) -> np.ndarray:
    """
    Load a thermal TIFF image.

    Parameters
    ----------
    image_path : Path
        Path to the TIFF image.

    Returns
    -------
    np.ndarray
        Original thermal image.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    return tifffile.imread(image_path)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize thermal image to 8-bit grayscale.

    Parameters
    ----------
    image : np.ndarray
        Original thermal image.

    Returns
    -------
    np.ndarray
        Normalized uint8 image.
    """

    normalized_image = cv2.normalize(
        image,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    return normalized_image


def prepare_display_image(image: np.ndarray) -> np.ndarray:
    """
    Convert grayscale image to RGB for visualization.

    Parameters
    ----------
    image : np.ndarray
        Normalized image.

    Returns
    -------
    np.ndarray
        RGB image suitable for plotting.
    """

    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    return image.copy()