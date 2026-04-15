"""Sample viewer for displaying clean vs augmented eye images and face mesh.

Creates combined visualizations showing eye images with zoom and face mesh
point overlays for inspection of data augmentation effects.
"""

from typing import Dict, Any
import numpy as np
import cv2
from Core.data.visualization_utils import (
    normalize_img,
    create_points_image,
    overlay_augmented_on_clean,
)

# Constants for visualization
FACE_MESH_VISUALIZATION_SIZE = 256  # Size of face mesh point visualization image
FIRST_TIMESTEP = 0  # Index of first timestep to visualize
HEADER_HEIGHT = 50  # Height of text header in visualization
TEXT_COLOR = (255, 255, 255)  # White color for text in BGR format
TEXT_FONT_SCALE = 1.0  # Font scale for text
TEXT_THICKNESS = 1  # Thickness of text lines
TEXT_X_POSITION = 20  # Horizontal position offset for text labels
TEXT_Y_POSITION = 35  # Vertical position for text labels
TEXT_COUNTER_X_OFFSET = 220  # Offset from right edge for sample counter


def create_visualization(X: Dict[str, Any], zoom_factor: int) -> np.ndarray:
    """Create combined visualization from sampled data.

    Args:
        X: Sample data dict with 'clean' and 'augmented' keys
        zoom_factor: Magnification factor for display

    Returns:
        Combined image with left/right eye images (clean and augmented) and face mesh point visualizations
    """
    # Extract eye images and points from samples
    clean_data = X["clean"]
    augmented_data = X["augmented"]

    # Extract eye images (shape: batch, timesteps, 32, 32, 1)
    clean_left_eye = clean_data["left eye"]
    clean_right_eye = clean_data["right eye"]
    augmented_left_eye = augmented_data["left eye"]
    augmented_right_eye = augmented_data["right eye"]

    # Extract face mesh points (shape: batch, timesteps, 478, 2)
    clean_points = clean_data["points"]
    augmented_points = augmented_data["points"]

    # Create combined visualization
    rows = []
    actual_samples = len(clean_left_eye)

    for idx in range(actual_samples):
        # Extract first timestep (remove timesteps and channel dimensions)
        clean_left_img = normalize_img(
            clean_left_eye[idx, FIRST_TIMESTEP, :, :, 0]
        )  # (32, 32)
        clean_right_img = normalize_img(
            clean_right_eye[idx, FIRST_TIMESTEP, :, :, 0]
        )  # (32, 32)
        augmented_left_img = normalize_img(
            augmented_left_eye[idx, FIRST_TIMESTEP, :, :, 0]
        )  # (32, 32)
        augmented_right_img = normalize_img(
            augmented_right_eye[idx, FIRST_TIMESTEP, :, :, 0]
        )  # (32, 32)

        # Extract points for this sample at first timestep
        clean_pts = clean_points[idx, FIRST_TIMESTEP, :, :]  # (478, 2)
        augmented_pts = augmented_points[idx, FIRST_TIMESTEP, :, :]  # (478, 2)

        # Resize eye images with zoom using nearest-neighbor interpolation
        h, w = clean_left_img.shape[:2]

        def zoom_image(img: np.ndarray) -> np.ndarray:
            """Resize image using nearest-neighbor interpolation.

            Args:
                img: Input image to zoom.

            Returns:
                Zoomed image with dimensions (h*zoom_factor, w*zoom_factor).
            """
            return cv2.resize(
                src=img,
                dsize=(w * zoom_factor, h * zoom_factor),
                interpolation=cv2.INTER_NEAREST,
            )

        zoomed_clean_left = zoom_image(clean_left_img)
        zoomed_clean_right = zoom_image(clean_right_img)
        zoomed_augmented_left = zoom_image(augmented_left_img)
        zoomed_augmented_right = zoom_image(augmented_right_img)

        # Convert grayscale to BGR for consistent visualization
        zoomed_clean_left = cv2.cvtColor(src=zoomed_clean_left, code=cv2.COLOR_GRAY2BGR)
        zoomed_clean_right = cv2.cvtColor(
            src=zoomed_clean_right, code=cv2.COLOR_GRAY2BGR
        )
        zoomed_augmented_left = cv2.cvtColor(
            src=zoomed_augmented_left, code=cv2.COLOR_GRAY2BGR
        )
        zoomed_augmented_right = cv2.cvtColor(
            src=zoomed_augmented_right, code=cv2.COLOR_GRAY2BGR
        )

        points_img_size = zoomed_augmented_left.shape[1]

        # Create face mesh point visualizations with same scale bounds
        # Use clean points to establish the scale
        clean_points_img, scale_bounds = create_points_image(
            clean_pts, img_size=points_img_size
        )
        # Overlay augmented points on clean with red lines
        overlay_points_img = overlay_augmented_on_clean(
            clean_points_img,
            clean_pts,
            augmented_pts,
            scale_bounds,
            img_size=points_img_size,
        )

        # Create row: [clean left | clean right | augmented left | augmented right | face mesh]
        row = np.hstack(
            [
                zoomed_clean_left,
                zoomed_clean_right,
                zoomed_augmented_left,
                zoomed_augmented_right,
                overlay_points_img,
            ]
        )
        rows.append(row)

    # Combine all rows vertically
    combined_img = np.vstack(rows)

    # Add column headers at the top
    header_height = HEADER_HEIGHT
    header = np.zeros((header_height, combined_img.shape[1], 3), dtype=np.uint8)
    combined_with_header = np.vstack([header, combined_img])

    # Add text labels for columns
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = TEXT_FONT_SCALE
    thickness = TEXT_THICKNESS
    color = TEXT_COLOR

    # Column positions (approximate, relative to image width)
    col_width = w * zoom_factor
    x_clean_left = TEXT_X_POSITION
    x_clean_right = col_width + TEXT_X_POSITION
    x_aug_left = col_width * 2 + TEXT_X_POSITION
    x_aug_right = col_width * 3 + TEXT_X_POSITION
    x_points = col_width * 4 + TEXT_X_POSITION

    # Eye image headers
    cv2.putText(
        img=combined_with_header,
        text=f"Clean Left ({h}x{w})",
        org=(x_clean_left, TEXT_Y_POSITION),
        fontFace=font,
        fontScale=font_scale,
        color=color,
        thickness=thickness,
    )
    cv2.putText(
        img=combined_with_header,
        text=f"Clean Right ({h}x{w})",
        org=(x_clean_right, TEXT_Y_POSITION),
        fontFace=font,
        fontScale=font_scale,
        color=color,
        thickness=thickness,
    )
    cv2.putText(
        img=combined_with_header,
        text="Augmented Left",
        org=(x_aug_left, TEXT_Y_POSITION),
        fontFace=font,
        fontScale=font_scale,
        color=color,
        thickness=thickness,
    )
    cv2.putText(
        img=combined_with_header,
        text="Augmented Right",
        org=(x_aug_right, TEXT_Y_POSITION),
        fontFace=font,
        fontScale=font_scale,
        color=color,
        thickness=thickness,
    )

    # Face mesh points header - clean (colored) overlaid with augmented (red)
    cv2.putText(
        img=combined_with_header,
        text="Face Mesh: Clean (colored) + Augmented (red)",
        org=(x_points, TEXT_Y_POSITION),
        fontFace=font,
        fontScale=font_scale,
        color=color,
        thickness=thickness,
    )

    # Add sample counter in top-right
    cv2.putText(
        img=combined_with_header,
        text="Sample",
        org=(combined_with_header.shape[1] - TEXT_COUNTER_X_OFFSET, TEXT_Y_POSITION),
        fontFace=font,
        fontScale=font_scale,
        color=color,
        thickness=thickness,
    )

    return combined_with_header
