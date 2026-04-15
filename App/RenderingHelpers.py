#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rendering helper methods for the eye-tracking application.

Provides helper functions for rendering prediction data, face mesh landmarks,
and debug information overlays.
"""
from typing import List, Optional, Tuple

import numpy as np
import pygame

from App.DrawingUtils import draw_object, draw_text, draw_text_list
from App.Utils import Colors


def render_predictions(
    display_surf: pygame.Surface,
    font: pygame.font.Font,
    history: List[np.ndarray],
    smoothed_prediction: np.ndarray,
    show_predictions: bool,
    text_position_x: int = 5,
    text_position_y: int = 5,
) -> None:
    """Render gaze prediction visualization.

    Draws prediction history and smoothed prediction as colored circles and lines
    on the display surface.

    Args:
        display_surf: Target display surface.
        font: Pygame font for text rendering.
        history: List of prediction positions (normalized [0,1]²).
        smoothed_prediction: Current smoothed gaze position (normalized [0,1]²).
        show_predictions: Whether to render predictions.
        text_position_x: X coordinate for position text (default: 5).
        text_position_y: Y coordinate for position text (default: 5).
    """
    if not show_predictions or len(history) == 0:
        return

    window_size = np.array(display_surf.get_size())
    positions = np.array(history) * window_size[None]
    positions = positions.astype(np.int32)

    for prev_point, next_point in zip(positions[:-1], positions[1:]):
        pygame.draw.line(display_surf, Colors.WHITE, prev_point, next_point, 2)
        draw_object(display_surf, pos=tuple(next_point), R=3, color=Colors.PURPLE)

    draw_object(display_surf, pos=tuple(positions[-1]), R=5, color=Colors.RED)

    smoothed_pos = np.multiply(smoothed_prediction, window_size).astype(np.int32)
    draw_object(display_surf, pos=tuple(smoothed_pos), R=5, color=Colors.BLACK)

    draw_text(
        font,
        display_surf,
        text=str(positions),
        pos=(text_position_x, text_position_y),
        color=Colors.BLACK,
    )


def render_info(
    display_surf: pygame.Surface,
    font: pygame.font.Font,
    dataset_total_samples: int,
    predictor_masks: Tuple[bool, bool, bool],
    fps: float,
    window_dimensions: np.ndarray,
    show_face_mesh: bool,
    face_mesh: Optional[np.ndarray],
    webcams_list: List[str],
    current_webcam: int,
    start_x: int = 5,
    start_y: int = 95,
    line_height: int = 25,
) -> None:
    """Render debug information overlay.

    Args:
        display_surf: Target display surface.
        font: Pygame font for text rendering.
        dataset_total_samples: Total number of samples collected.
        predictor_masks: Tuple of (mask_face, mask_left_eye, mask_right_eye).
        fps: Current frames per second.
        window_dimensions: Window width and height as array.
        show_face_mesh: Whether to show face mesh landmarks.
        face_mesh: Face mesh coordinates (normalized) or None.
        webcams_list: List of available webcam names.
        current_webcam: Currently selected webcam index.
        start_x: Starting X position (default: 5).
        start_y: Starting Y position (default: 95).
        line_height: Vertical spacing between text lines (default: 25).
    """
    start_points = (start_x, start_y)
    texts: List[Tuple[str, Tuple[int, int, int]]] = []

    # Add sample count
    texts.append((f"Samples: {dataset_total_samples}", Colors.RED))

    # Add predictor mask status
    mask_face, mask_left_eye, mask_right_eye = predictor_masks
    modes = []
    if mask_face:
        modes.append("no face")
    if mask_left_eye:
        modes.append("no left eye")
    if mask_right_eye:
        modes.append("no right eye")

    if len(modes) > 0:
        texts.append((", ".join(modes), Colors.GREEN))

    # Add FPS
    texts.append((f"FPS: {fps:.1f}", Colors.BLACK))

    # Add resolution
    texts.append(
        (
            f"Resolution: {int(window_dimensions[0])} x {int(window_dimensions[1])}",
            Colors.BLACK,
        )
    )

    # Draw face mesh if enabled
    if show_face_mesh and face_mesh is not None:
        scaled = np.multiply(face_mesh, window_dimensions[None])
        scaled = scaled.astype(np.int32)
        for p in scaled:
            pygame.draw.circle(display_surf, Colors.RED, tuple(p), 2, 0)

    # Add webcam info
    texts.append(("", Colors.BLACK))
    texts.append(("Webcams:", Colors.BLACK))
    for i, name in enumerate(webcams_list):
        color = Colors.RED if i == current_webcam else Colors.BLACK
        texts.append((f"{i}: {name}", color))

    draw_text_list(font, display_surf, texts, start_points, line_height)
