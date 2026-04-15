"""Visualization utilities for face mesh points and eye images.

Provides functions for rendering face mesh landmarks, creating point overlays,
and normalizing images for display.
"""

from typing import Optional, Tuple
import numpy as np
import cv2
from Core.landmarks import FACE_MESH_INVALID_VALUE

# Constants for visualization
VISUALIZATION_PADDING = 0.05  # Padding fraction for point visualization
HUE_RANGE_DEGREES = 180  # HSV hue range in degrees (0-180 in OpenCV)
HSV_SATURATION = 255  # Maximum HSV saturation value
HSV_VALUE = 255  # Maximum HSV value (brightness)
MIN_RANGE_EPSILON = 1e-6  # Minimum range to prevent division by zero


def normalize_img(img: np.ndarray) -> np.ndarray:
    """Normalize image to 0-255 uint8 range.

    Converts float images (0-1 range) to uint8 (0-255 range), and ensures
    output is always uint8 dtype.

    Args:
        img: Input image as float32/float64 (0-1 range) or any numpy array.

    Returns:
        Image as uint8 (0-255 range).
    """
    if img.dtype == np.float32 or img.dtype == np.float64:
        return (np.clip(img, 0, 1) * 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        return img.astype(np.uint8)
    return img


def scale_points_to_pixels(
    points: np.ndarray, img_size: int, scale_bounds: Tuple[float, float, float, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert normalized face mesh points to pixel coordinates using scale bounds.

    Args:
        points: Face mesh points (478, 2) in normalized [0.0, 1.0] coordinates
        img_size: Size of output visualization image
        scale_bounds: Tuple of (min_x, min_y, max_x, max_y) for scaling

    Returns:
        Array of pixel coordinates and valid indices
    """
    # Filter valid points
    valid_mask = np.all(points != FACE_MESH_INVALID_VALUE, axis=-1)
    valid_indices = np.where(valid_mask)[0]
    valid_points = points[valid_mask]

    min_x, min_y, max_x, max_y = scale_bounds
    range_x = max(max_x - min_x, MIN_RANGE_EPSILON)
    range_y = max(max_y - min_y, MIN_RANGE_EPSILON)

    # Scale to fill image with padding
    padding = VISUALIZATION_PADDING
    if len(valid_points) > 0:
        scaled_x = (valid_points[:, 0] - min_x) / range_x * (1 - 2 * padding) + padding
        scaled_y = (valid_points[:, 1] - min_y) / range_y * (1 - 2 * padding) + padding

        pixel_points = np.column_stack(
            [(scaled_x * img_size).astype(int), (scaled_y * img_size).astype(int)]
        )
    else:
        pixel_points = np.array([], dtype=int).reshape(0, 2)

    return pixel_points, valid_indices


def get_hue_color(point_idx: int, num_points: int) -> Tuple[int, int, int]:
    """Get HSV-based color for a point index.

    Maps point index to a hue value and returns corresponding BGR color
    for visualization.

    Args:
        point_idx: Index of point for color assignment.
        num_points: Total number of points (for hue distribution).

    Returns:
        Tuple of (B, G, R) color values for OpenCV.
    """
    hue = int((point_idx / max(num_points, 1)) * HUE_RANGE_DEGREES)
    # OpenCV cvtColor works with uint8 arrays; numpy typing has limitations
    hsv_color = np.array([[[hue, HSV_SATURATION, HSV_VALUE]]], dtype=np.uint8)  # type: ignore[arg-type]
    # cv2.cvtColor returns ndarray which supports indexing despite type stub
    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0, 0]  # type: ignore[index]
    return (int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2]))


def create_points_image(
    points: np.ndarray,
    img_size: int = 256,
    point_size: int = 2,
    scale_bounds: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """Create visualization of face mesh points with unique colors.

    Args:
        points: Face mesh points (478, 2) in normalized [0.0, 1.0] coordinates
        img_size: Size of output visualization image (default: 256x256)
        point_size: Radius of drawn points (default: 2 pixels)
        scale_bounds: Tuple of (min_x, min_y, max_x, max_y) for consistent scaling.
                      If None, computed from points. Default: None

    Returns:
        Tuple of (image, bounds) where image has face mesh points drawn as circles,
        and bounds is (min_x, min_y, max_x, max_y) used for scaling
    """
    # Create black background
    points_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    # Determine scale bounds
    if scale_bounds is None:
        valid_mask = np.all(points != FACE_MESH_INVALID_VALUE, axis=-1)
        valid_points = points[valid_mask]
        if len(valid_points) > 0:
            min_x, min_y = valid_points.min(axis=0)
            max_x, max_y = valid_points.max(axis=0)
            scale_bounds = (min_x, min_y, max_x, max_y)
        else:
            scale_bounds = (0.0, 0.0, 1.0, 1.0)

    # Get pixel coordinates
    pixel_points, valid_indices = scale_points_to_pixels(points, img_size, scale_bounds)
    num_valid_points = len(valid_indices)

    # Draw points with unique colors
    for i, pt in enumerate(pixel_points):
        if 0 <= pt[0] < img_size and 0 <= pt[1] < img_size:
            bgr_color = get_hue_color(i, num_valid_points)
            cv2.circle(
                img=points_img,
                center=tuple(pt),
                radius=point_size,
                color=bgr_color,
                thickness=-1,
            )

    return points_img, scale_bounds


def overlay_augmented_on_clean(
    clean_points_img: np.ndarray,
    clean_points: np.ndarray,
    augmented_points: np.ndarray,
    scale_bounds: Tuple[float, float, float, float],
    img_size: int = 256,
    point_size: int = 2,
) -> np.ndarray:
    """Overlay augmented points on clean points image with red connecting lines.

    Args:
        clean_points_img: Image with clean points already drawn
        clean_points: Clean face mesh points (478, 2)
        augmented_points: Augmented face mesh points (478, 2)
        scale_bounds: Scale bounds from clean points
        img_size: Size of visualization image
        point_size: Radius of drawn points

    Returns:
        Image with augmented points overlaid as red circles and red lines to clean points.
        Clean points shown as white if augmented version doesn't exist (dropout).
    """
    result_img = clean_points_img.copy()

    # Get pixel coordinates for both clean and augmented
    clean_pixel_points, clean_valid_indices = scale_points_to_pixels(
        clean_points, img_size, scale_bounds
    )
    aug_pixel_points, aug_valid_indices = scale_points_to_pixels(
        augmented_points, img_size, scale_bounds
    )

    # Create sets for quick lookup of valid point indices
    aug_valid_set = set(aug_valid_indices)

    red_color = (0, 0, 255)  # Red in BGR
    white_color = (255, 255, 255)  # White in BGR for missing augmented points
    line_thickness = 1

    # First pass: draw red lines and red circles for existing augmented points
    for i, (clean_idx, aug_idx) in enumerate(
        zip(clean_valid_indices, aug_valid_indices)
    ):
        if clean_idx == aug_idx:
            clean_pt = clean_pixel_points[i]
            aug_pt = aug_pixel_points[i]
            # Draw line from clean to augmented point
            cv2.line(
                img=result_img,
                pt1=tuple(clean_pt),
                pt2=tuple(aug_pt),
                color=red_color,
                thickness=line_thickness,
            )
            # Draw augmented point as red circle
            cv2.circle(
                img=result_img,
                center=tuple(aug_pt),
                radius=point_size,
                color=red_color,
                thickness=-1,
            )

    # Second pass: draw white circles for clean points that have no augmented version (dropout)
    # Draw with larger radius to be visible over the colored clean points
    dropout_radius = point_size + 1
    for i, clean_idx in enumerate(clean_valid_indices):
        # Check if this clean point doesn't have a corresponding augmented point
        if clean_idx not in aug_valid_set:
            clean_pt = clean_pixel_points[i]
            # Draw white circle outline (indicates dropout) - larger to be visible
            cv2.circle(
                img=result_img,
                center=tuple(clean_pt),
                radius=dropout_radius,
                color=white_color,
                thickness=1,
            )

    return result_img
