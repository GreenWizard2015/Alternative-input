# Draw random illumination on the pygame screen.
import pygame
import numpy as np
import scipy.interpolate as sInterp
import time
import random

class CIlluminationSource:
  def __init__(self):
    self._radius = 310
    self._color = np.random.random((3, ))
    self._points = None
    self._newSpline(extend=False)
    return
  
  def _newSpline(self, extend=True):
    self._T = 0.0
    N = 3
    points = np.random.normal(size=(4, 2), loc=0.5, scale=0.5)
    if extend:
      points = np.concatenate([self._points[-N:], points], axis=0)
    
    self._points = points = np.clip(points, -0.5, 1.5)
    distance = np.cumsum( np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=-1 )) )
    distance = np.insert(distance, 0, 0)
    
    speed = np.random.uniform(1.0, 4.0, size=1)[0]
    T = distance[-1] / speed
    self._maxT = np.clip(T, 20, 40)
    distance /= distance[-1]
    
    shift = distance[N - 1] if extend else 0.0
    splines = [sInterp.CubicSpline(distance, coords) for coords in points.T]
    self._getPoint = lambda t: np.array(
      [s((t * (1 - shift)) + shift) for s in splines]
    )
    return
  
  def on_tick(self, deltaT):
    self._T += deltaT
    if self._maxT < self._T: self._newSpline()
    
    pos = self._getPoint(self._T / self._maxT)
    self._pos = np.clip(pos, 0.0, 1.0)
    return
  
  def on_render(self, window):
    # render the light source
    wh = np.array(window.get_size())
    pos = (self._pos * wh).astype(np.int32)
    color = (self._color * 255).astype(np.int32)
    pygame.draw.circle(window, color, pos, self._radius)
    return
  pass

class CRandomIllumination:
  def __init__(self, sourcesN=32):
    self._sources = [CIlluminationSource() for i in range(sourcesN)]
    return
  
  def on_tick(self, deltaT):
    for source in self._sources:
      source.on_tick(deltaT)
    return

  def on_render(self, window):
    for source in self._sources:
      source.on_render(window)
    return
  pass