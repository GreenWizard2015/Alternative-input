import pygame as G
import numpy as np

class CAppMode:
  def __init__(self, app):
    self._app = app
    self._paused = True
    return
  
  def on_event(self, event):
    if event.type == G.KEYDOWN:
      if event.key in [G.K_p, G.K_RETURN]:
        self._paused = not self._paused

      if event.key == G.K_SPACE:
        self._paused = True
    return
  
  def on_render(self, window):
    return
  
  def on_sample(self, tracked):
    if self._paused: return
    self._app._dataset.store(tracked, np.array(self._pos))
    return
  
  def on_prediction(self, pos, data):
    return
  
  @property
  def paused(self): return self._paused
  pass
