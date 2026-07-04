"""
masks.py
---------
Functions for generating masks used during hotspot detection.
"""

import cv2
import numpy as np


def create_panel_mask(
    image_shape: tuple,
    panel_polygon: list
) -> np.ndarray:
    """
    Create a binary mask representing the solar panel.

    Parameters
    ----------
    image_shape : tuple
        Shape of the grayscale image.

    panel_polygon : list
        Coordinates of panel corners.

    Returns
    -------
    np.ndarray
        Binary panel mask.
    """

    mask = np.zeros(image_shape, dtype=np.uint8)

    polygon = np.array(
        [panel_polygon],
        dtype=np.int32
    )

    cv2.fillPoly(
        mask,
        polygon,
        255
    )

    return mask


def draw_panel_boundary(
    image: np.ndarray,
    panel_polygon: list
) -> np.ndarray:
    """
    Draw panel outline on image.

    Parameters
    ----------
    image : np.ndarray

    panel_polygon : list

    Returns
    -------
    np.ndarray
    """

    output = image.copy()

    polygon = np.array(
        [panel_polygon],
        dtype=np.int32
    )

    cv2.polylines(
        output,
        polygon,
        True,
        (255, 0, 0),
        3
    )

    return output

def create_inner_mask(
    panel_mask: np.ndarray,
    erosion_kernel_size: tuple
) -> np.ndarray:
    """
    Create an inner mask by eroding the panel mask.
    """

    kernel = np.ones(
        erosion_kernel_size,
        dtype=np.uint8
    )

    inner_mask = cv2.erode(
        panel_mask,
        kernel,
        iterations=1
    )

    return inner_mask


def remove_separator_lines(
    inner_mask: np.ndarray
) -> np.ndarray:
    """
    Remove horizontal separator lines inside the panel.
    """

    mask = inner_mask.copy()

    height, width = mask.shape

    separator_lines = [
        int(height * 0.33),
        int(height * 0.66)
    ]

    for y in separator_lines:

        cv2.rectangle(
            mask,
            (0, y - 12),
            (width, y + 12),
            0,
            -1
        )

    return mask


def apply_exclusion_region(
    mask: np.ndarray,
    exclusion_polygon: list
) -> np.ndarray:
    """
    Remove unwanted diagonal region.
    """

    output_mask = mask.copy()

    polygon = np.array(
        [exclusion_polygon],
        dtype=np.int32
    )

    cv2.fillPoly(
        output_mask,
        polygon,
        0
    )

    return output_mask