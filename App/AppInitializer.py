#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Application initialization and state management helpers.

Provides helper functions for initializing application display,
setting up camera views, and managing predictor masks.
"""
from typing import List, Tuple

import numpy as np
import pygame

from cv2_enumerate_cameras import enumerate_cameras


def initialize_pygame_display() -> Tuple[pygame.Surface, pygame.font.Font]:
    """Initialize pygame display in fullscreen mode.

    Returns:
        Tuple of (display_surface, font_object).

    Note:
        Initializes pygame if not already initialized.
    """
    pygame.init()

    info = pygame.display.Info()
    w = info.current_w
    h = info.current_h
    pygame.display.set_mode((w, h), pygame.FULLSCREEN)

    pygame.display.set_caption("App")
    font = pygame.font.Font(pygame.font.get_default_font(), 16)

    return pygame.display.get_surface(), font


def create_camera_view(
    camera_view_x: int, camera_view_y: int, camera_view_size: int
) -> Tuple[np.ndarray, pygame.Surface]:
    """Create camera view display configuration.

    Args:
        camera_view_x: X position for camera view.
        camera_view_y: Y position for camera view.
        camera_view_size: Size of camera view in pixels.

    Returns:
        Tuple of (view_coordinates, surface).
    """
    camera_view = np.array(
        [
            (camera_view_x, camera_view_y),
            (camera_view_x + camera_view_size, camera_view_y + camera_view_size),
        ]
    )
    camera_surface = pygame.Surface(camera_view[1] - camera_view[0])
    return camera_view, camera_surface


def create_eyes_view(
    camera_view_x: int,
    camera_view_y: int,
    camera_view_size: int,
    eyes_y_offset: int,
    eyes_height: int,
) -> Tuple[np.ndarray, pygame.Surface]:
    """Create eyes region view display configuration.

    Args:
        camera_view_x: X position for camera view.
        camera_view_y: Y position for camera view.
        camera_view_size: Size of camera view in pixels.
        eyes_y_offset: Y offset from camera view for eyes region.
        eyes_height: Height of eyes region in pixels.

    Returns:
        Tuple of (view_coordinates, surface).
    """
    eyes_y = camera_view_y + camera_view_size + eyes_y_offset
    eyes_view = np.array(
        [
            (camera_view_x, eyes_y),
            (camera_view_x + camera_view_size, eyes_y + eyes_height),
        ]
    )
    eyes_surface = pygame.Surface(eyes_view[1] - eyes_view[0])
    return eyes_view, eyes_surface


def get_available_webcams() -> List[str]:
    """Get list of available webcams with their names.

    Returns:
        List of strings in format "index: name".
    """
    webcams_list = []
    for camera_info in enumerate_cameras():
        webcams_list.append(f"{camera_info.index}: {camera_info.name}")
    return webcams_list
