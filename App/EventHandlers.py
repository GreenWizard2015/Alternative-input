#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Event handling utilities for the application.

Provides helper functions for processing pygame events,
including key press handling and mode switching.
"""
from typing import Callable

import pygame.locals as G


def handle_key_down_event(
    event_key: int,
    app_modes_count: int,
    on_toggle_predictions: Callable,
    on_toggle_illumination: Callable,
    on_toggle_face_mask: Callable,
    on_toggle_left_eye_mask: Callable,
    on_toggle_right_eye_mask: Callable,
    on_mode_change: Callable,
) -> None:
    """Handle keyboard key down events.

    Args:
        event_key: The key code from the pygame event.
        app_modes_count: Number of available application modes.
        on_toggle_predictions: Callback to toggle predictions display (S key).
        on_toggle_illumination: Callback to toggle illumination (L key).
        on_toggle_face_mask: Callback to toggle face mask (F1 key).
        on_toggle_left_eye_mask: Callback to toggle left eye mask (F2 key).
        on_toggle_right_eye_mask: Callback to toggle right eye mask (F3 key).
        on_mode_change: Callback for mode change (1-9 keys), receives mode index.
    """
    # Escape key: quit
    if event_key == G.K_ESCAPE:
        return

    # S key: toggle predictions
    if event_key == G.K_s:
        on_toggle_predictions()
        return

    # 1-9 keys: switch mode
    if (G.K_1 <= event_key) and (event_key < (G.K_1 + app_modes_count)):
        mode_index = event_key - G.K_1
        on_mode_change(mode_index)
        return

    # L key: toggle illumination
    if event_key == G.K_l:
        on_toggle_illumination()
        return

    # F1 key: toggle face mask
    if event_key == G.K_F1:
        on_toggle_face_mask()
        return

    # F2 key: toggle left eye mask
    if event_key == G.K_F2:
        on_toggle_left_eye_mask()
        return

    # F3 key: toggle right eye mask
    if event_key == G.K_F3:
        on_toggle_right_eye_mask()
        return
