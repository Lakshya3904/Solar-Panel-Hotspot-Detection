"""
hotspot_detector.py
-------------------
Functions for hotspot detection.
"""

import numpy as np


def calculate_hotspot_threshold(
    gray_image: np.ndarray,
    roi_mask: np.ndarray,
    percentile: float
) -> float:
    """
    Calculate hotspot threshold from ROI pixels.

    Parameters
    ----------
    gray_image : np.ndarray
        Grayscale thermal image.

    roi_mask : np.ndarray
        Region of interest mask.

    percentile : float
        Percentile used for thresholding.

    Returns
    -------
    float
        Threshold value.
    """

    panel_pixels = gray_image[roi_mask > 0]

    threshold = np.percentile(
        panel_pixels,
        percentile
    )

    return threshold


def generate_hotspot_mask(
    gray_image: np.ndarray,
    roi_mask: np.ndarray,
    threshold: float
) -> np.ndarray:
    """
    Generate binary hotspot mask.

    Parameters
    ----------
    gray_image : np.ndarray

    roi_mask : np.ndarray

    threshold : float

    Returns
    -------
    np.ndarray
        Binary hotspot mask.
    """

    hotspot_mask = np.zeros_like(
        gray_image,
        dtype=np.uint8
    )

    hotspot_mask[
        (gray_image >= threshold)
        &
        (roi_mask > 0)
    ] = 255

    return hotspot_mask