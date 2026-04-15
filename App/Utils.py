"""UI rendering utilities for pygame-based visualization.

Provides utilities for color management, image conversion (OpenCV to pygame),
vector operations, and real-time surface pixel manipulation.

Functions:
    - normalized: L-p norm normalization
    - densityToSurface: RGB array to pygame Surface
    - cv2ImageToSurface: OpenCV image conversion with format handling
    - rotate: 2D vector rotation
    - numpyToSurfaceBind: Direct pixel buffer binding for fast updates

Classes:
    - Colors: Standard color palette (RGB tuples)
"""

import cv2
import pygame
import numpy as np
from typing import List, Tuple, Literal


class Colors:
    """Standard color palette with RGB tuples.

    Contains standard web colors (BLACK, WHITE, RED, etc.) and matplotlib
    named colors (dynamically loaded). All colors are represented as RGB
    tuples (R, G, B) with values in range 0-255.

    Attributes:
        BLACK, WHITE, BLUE, GREEN, RED, PURPLE, SILVER: Standard colors
        asList: List of all available RGB color tuples
    """

    BLACK = (0, 0, 0)
    SILVER = (192, 192, 192)
    WHITE = (255, 255, 255)
    BLUE = (0, 0, 255)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    PURPLE = (255, 0, 255)
    asList: List[Tuple[int, int, int]] = []  # Populated by _makeColors()


def _makeColors() -> None:
    """Populate Colors class with standard matplotlib color names.

    Extracts all named colors from matplotlib and adds them as uppercase
    attributes to the Colors class with RGB tuples.
    """
    import matplotlib.colors as mcolors

    for name, hex_color in mcolors.cnames.items():
        hex_str: str = str(hex_color)
        setattr(
            Colors, name.upper(), tuple(int(hex_str[i : i + 2], 16) for i in (1, 3, 5))
        )


_makeColors()
# Add all colors to a list if they are RGB tuples (3-element tuples from __dict__)
Colors.asList = [
    rgb for rgb in Colors.__dict__.values() if isinstance(rgb, tuple) and len(rgb) == 3
]


def normalized(a: np.ndarray, axis: int = -1, order: int = 2) -> np.ndarray:
    """Normalize array along specified axis using vector norm.

    Divides array by its L-p norm to produce unit vectors.
    Avoids division by zero by replacing zero norms with 1.

    Args:
        a: Input array to normalize
        axis: Axis along which to compute norm (default: -1)
        order: Norm order (1, 2, inf, etc.) (default: 2 for L2 norm)

    Returns:
        Normalized array with same shape as input.

    Example:
        >>> arr = np.array([[1, 0], [0, 1]])
        >>> normalized(arr)
        array([[1., 0.],
               [0., 1.]])
    """
    norm: np.ndarray = np.atleast_1d(np.linalg.norm(x=a, ord=order, axis=axis))
    norm[norm == 0] = 1
    return a / np.expand_dims(norm, axis=axis)


def densityToSurface(cv2Image: np.ndarray) -> pygame.Surface:
    """Convert RGB array to pygame Surface.

    Converts a numpy array (H, W, 3) representing RGB density/heatmap
    to a pygame Surface for rendering.

    Args:
        cv2Image: RGB array of shape (height, width, 3)

    Returns:
        Converted pygame Surface
    """
    size = cv2Image.shape[:-1]
    fmt: Literal["RGB", "RGBA", "BGRA"] = "RGB"
    img_bytes: bytes = cv2Image.tobytes()
    surface = pygame.image.frombuffer(img_bytes, size=size, format=fmt)
    return surface.convert()


def cv2ImageToSurface(cv2Image: np.ndarray) -> pygame.Surface:
    """Convert OpenCV image to pygame Surface.

    Handles uint16 depth conversion, grayscale expansion, BGR-to-RGB conversion,
    and RGBA alpha channel support.

    Args:
        cv2Image: OpenCV image (uint8 or uint16, grayscale or BGR/BGRA)

    Returns:
        Converted pygame Surface with alpha support if RGBA
    """
    if cv2Image.dtype.name == "uint16":
        cv2Image = (cv2Image / 256).astype("uint8")

    size = cv2Image.shape[1::-1]
    fmt: Literal["RGB", "RGBA", "BGRA"] = "RGB"
    if len(cv2Image.shape) == 2:
        cv2Image = np.repeat(cv2Image.reshape(size[1], size[0], 1), repeats=3, axis=2)
        fmt = "RGB"
    else:
        fmt = "RGBA" if cv2Image.shape[2] == 4 else "RGB"
        cv2Image[:, :, [0, 2]] = cv2Image[:, :, [2, 0]]

    img_bytes: bytes = cv2Image.tobytes()
    surface = pygame.image.frombuffer(img_bytes, size=size, format=fmt)
    return surface.convert_alpha() if fmt == "RGBA" else surface.convert()


def rotate(vector: np.ndarray, rads: float) -> np.ndarray:
    """Rotate 2D vector by angle in radians.

    Applies 2D rotation matrix: R(θ) = [[cos, -sin], [sin, cos]].

    Args:
        vector: 2D vector [x, y]
        rads: Rotation angle in radians

    Returns:
        Rotated 2D vector of same shape
    """
    return np.array(
        [
            np.cos(rads) * vector[0] - np.sin(rads) * vector[1],
            np.sin(rads) * vector[0] + np.cos(rads) * vector[1],
        ]
    )


def numpyToSurfaceBind(array: np.ndarray, surface: pygame.Surface) -> None:
    """Bind numpy array directly to pygame Surface pixel data.

    Resizes array to match surface dimensions and updates surface pixels in-place.
    Handles grayscale-to-RGB conversion and axis swapping for pygame format.

    Args:
        array: Input array (grayscale or RGB) to bind to surface.
        surface: pygame Surface to update with array pixel data.

    Note:
        Modifies surface in-place. Releases reference after update.

    Example:
        >>> surface = pygame.Surface((640, 480))
        >>> array = np.zeros((480, 640, 3), dtype=np.uint8)
        >>> numpyToSurfaceBind(array, surface)
    """
    surf = pygame.surfarray.pixels3d(surface)
    wh: Tuple[int, int] = (surf.shape[0], surf.shape[1])
    array = cv2.resize(src=array, dsize=wh)
    grayscale_dims = 2
    rgb_channels = 1
    if grayscale_dims == len(array.shape):
        array = array.reshape((*array.shape, 1))  # H x W -> H x W x 1
    if rgb_channels == array.shape[-1]:
        array = np.repeat(array, repeats=3, axis=-1)  # grayscale -> RGB
    array = np.swapaxes(array, axis1=0, axis2=1)  # H x W x C -> W x H x C
    surf[:, :, :] = array
    del surf  # release surface
