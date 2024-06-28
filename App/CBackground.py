import pygame
import pygame.locals as G
import numpy as np
import time
from App.Utils import Colors

class CBackground:
  def __init__(self):
    self._backgroundDynamic = False
    return
  
  def on_tick(self, deltaT):
    return
  
  def _brightness(self):
    T = time.time()
    amplitude = 1.0 / 2.0
    duration = 30.0
    # smooth brightness change over N seconds
    sin = np.sin(2.0 * np.pi * T / duration)
    res = 1.0 + amplitude * sin
    return res

  def on_render(self, window):
    bg = Colors.SILVER
    if self._backgroundDynamic:
      # take color from Colors.asList based on current time, change every 5 seconds
      bg = Colors.asList[int(time.time() / 5) % len(Colors.asList)]
    # apply brightness
    bg = np.multiply(bg, self._brightness()).clip(0, 255).astype(np.uint8)
    window.fill(bg)
    return
  
  def on_event(self, event):
    if not(event.type == G.KEYDOWN): return
    # toggle background dynamic (B)
    if G.K_b == event.key:
      self._backgroundDynamic = not self._backgroundDynamic
    return
  pass