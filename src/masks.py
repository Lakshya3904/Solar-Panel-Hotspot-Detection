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