#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Drawing utility functions for pygame rendering.

Provides text and object drawing functionality for the application UI,
including text rendering with scaling and positioning, circular objects,
and target visualization with animated rings.
"""
import time
from typing import List, Tuple

import numpy as np
import pygame

from App.Utils import Colors


def draw_text(
    font: pygame.font.Font,
    display_surf: pygame.Surface,
    text: str,
    pos: Tuple[int, int],
    color: Tuple[int, int, int],
    scale: float = 1.0,
    center: bool = False,
) -> None:
    """Draw text to the display surface.

    Args:
        font: Pygame font object for rendering.
        display_surf: Target display surface.
        text: Text string to render.
        pos: (x, y) position for text.
        color: RGB color tuple.
        scale: Text scale factor (default: 1.0).
        center: Whether to center text at position (default: False).
    """
    text_surface = font.render(text, False, color)
    if scale != 1.0:
        text_surface = pygame.transform.scale(
            text_surface,
            (
                int(text_surface.get_width() * scale),
                int(text_surface.get_height() * scale),
            ),
        )

    if center:
        pos = tuple(np.subtract(pos, np.divide(text_surface.get_size(), 2)))

    pos_ints: Tuple[int, int] = (int(pos[0]), int(pos[1]))
    display_surf.blit(text_surface, pos_ints)


def draw_text_list(
    font: pygame.font.Font,
    display_surf: pygame.Surface,
    texts: List[Tuple[str, Tuple[int, int, int]]],
    start: Tuple[int, int],
    height: int,
) -> None:
    """Draw a list of text strings with specified spacing.

    Args:
        font: Pygame font object for rendering.
        display_surf: Target display surface.
        texts: List of (text_string, color) tuples.
        start: Starting (x, y) position.
        height: Vertical spacing between text lines in pixels.
    """
    x_pos, y_pos = start
    for text, color in texts:
        draw_text(font, display_surf, text=text, pos=(x_pos, y_pos), color=color)
        y_pos += height


def draw_object(
    display_surf: pygame.Surface,
    pos: Tuple[int, int],
    R: int = 10,
    color: Tuple[int, int, int] = Colors.WHITE,
) -> None:
    """Draw a circle or target object.

    Args:
        display_surf: Target display surface.
        pos: (x, y) center position.
        R: Radius in pixels (default: 10).
        color: RGB color tuple (default: WHITE).
    """
    if np.all(np.equal(color, Colors.WHITE)):
        draw_target(display_surf, pos=pos, R=R)
    else:
        pygame.draw.circle(display_surf, color, pos, R, 0)


def draw_target(
    display_surf: pygame.Surface,
    pos: Tuple[int, int],
    R: int = 10,
    target_draw_step: int = 2,
    circle_width_multiplier: int = 7,
) -> None:
    """Draw a target with crosshairs and rotating ring.

    Args:
        display_surf: Target display surface.
        pos: (x, y) center position.
        R: Radius in pixels (default: 10).
        target_draw_step: Step size for drawing concentric circles (default: 2).
        circle_width_multiplier: Multiplier for circle width animation (default: 7).
    """
    T = int(time.time())
    colors = Colors.asList
    for i in reversed(range(target_draw_step, R, target_draw_step)):
        color = colors[(i * circle_width_multiplier + T) % len(colors)]
        pygame.draw.circle(display_surf, color, pos, i, 0)
    # Draw contrast borders
    pygame.draw.circle(display_surf, Colors.BLACK, pos, R, 1)
    pygame.draw.circle(display_surf, Colors.WHITE, pos, R + 1, 1)
    pygame.draw.circle(display_surf, Colors.RED, pos, R + 2 + 3, 3)
