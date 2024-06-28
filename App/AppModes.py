import numpy as np
import pygame.locals as G
from App.Utils import Colors, normalized
from App.CSpinningTarget import CSpinningTarget
import time
import scipy.interpolate as sInterp
from App.CGameMode import CGameMode
from App.CAppMode import CAppMode

class CMoveToGoal(CAppMode):
  def __init__(self, app):
    super().__init__(app)
    self._speed = 55 * 2 * 2
    self._pos = self._goal = np.zeros((2, )) + 0.5
    return
  
  def on_tick(self, deltaT):
    wh = self._app.WH
    pos = np.multiply(wh, self._pos)
    goal = np.multiply(wh, self._goal)
    
    vec = normalized(np.subtract(goal, pos))[0]
    self._pos = np.add(pos, vec * self._speed * deltaT) / wh

    dist = np.sqrt(np.square(np.subtract(pos, goal)).sum())
    if dist < 3.0:
      self._goal = self._nextGoal(self._goal)
    return
  
  def on_render(self, window):
    super().on_render(window)
    
    wh = np.array(window.get_size())
    pos = tuple(int(x) for x in np.multiply(wh, self._pos))  
    self._app.drawObject(pos)
    return
  pass

class CCircleMovingMode(CMoveToGoal):
  def __init__(self, app):
    super().__init__(app)
    self._transitionT = 2
    self._maxLevel = 25
    self._level = 0
    self._reset()
    return
  
  def _nextGoal(self, old):
    if self._transitionStart is None:
      self._transitionStart = time.time()
    if (time.time() - self._transitionStart) < self._transitionT: return old
    
    self._transitionStart = None
    if len(self._path) <= 0:
      self._reset()
      return self._goal
    
    goal, *self._path = self._path
    return goal
  
  def on_event(self, event):
    super().on_event(event)
    if event.type == G.KEYDOWN:
      if G.K_UP == event.key:
        self._level = min((self._maxLevel, self._level + 1))
        self._reset()
        return
      
      if G.K_DOWN == event.key:
        self._level = max((0, self._level - 1))
        self._reset()
        return
      
      if G.K_RIGHT == event.key:
        self._reset(clockwise=False)
        self._active = True
        return
      
      if G.K_LEFT == event.key:
        self._reset(clockwise=True)
        self._active = True
        return
    return

  def on_sample(self, tracked):
    if self._active:
      super().on_sample(tracked)
    return
  
  def _reset(self, clockwise=False):
    path = np.array([
      [-1,  1],
      [ 1,  1],
      [ 1, -1],
      [-1, -1],
      [-1,  1],
    ], np.float32)
    lvl = (self._maxLevel - self._level) / ((2.0 * self._maxLevel) + 0)
    path = 0.5 + lvl * path
    if clockwise: path = path[::-1]
    self._pos, self._goal, *self._path = path
    self._active = False
    self._transitionStart = None
    return
  
  def on_tick(self, deltaT):
    if self._active:
      super().on_tick(deltaT)
    return
  
class CLookAtMode(CAppMode):
  def __init__(self, app):
    super().__init__(app)
    self._visibleT = 5.0
    self._next()
    return
  
  def _next(self):
    self._pos = np.random.uniform(size=(2,))
    self._active = False
    self._startT = None
    return
  
  def on_event(self, event):
    super().on_event(event)
    if event.type == G.KEYDOWN:
      if G.K_RIGHT == event.key:
        self._active = True
    return
  
  def on_tick(self, deltaT):
    if self._active:
      dT = time.time() - self._startT
      if self._visibleT < dT: self._next()
    else:
      self._startT = time.time()
    return
  
  def on_sample(self, tracked):
    if self._active:
      super().on_sample(tracked)
    return
  
  def on_render(self, window):
    super().on_render(window)
    wh = np.array(window.get_size())
    pos = tuple(int(x) for x in np.multiply(wh, self._pos))
    
    if self._active:
      self._app.drawObject(pos, color=Colors.RED)
    else:
      self._app.drawObject(pos, color=Colors.WHITE)
    return
  pass
###############################
class CSplineMode(CAppMode):
  def __init__(self, app):
    super().__init__(app)
    self._pos = np.zeros((2, )) + 0.5
    self._target = CSpinningTarget(app)
    self._points = None
    self._scale = 1.0
    self._newSpline(extend=False)
    return

  def _updateScale(self):
    newScale = np.random.uniform(0.1, 0.2) + self._scale
    self._scale = newScale
    if 1.0 < self._scale: self._scale = 0.0
    return newScale
  
  def _newSpline(self, extend=True):
    self._T = 0.0
    N = 3
    scale = self._updateScale()
    points = np.random.uniform(size=(N + 1, 2)) - 0.5
    points /= np.linalg.norm(points, axis=-1, keepdims=True) + 1e-6
    points = 0.5 + (points * scale)
    if extend:
      points = np.concatenate([self._points[-N:], points], axis=0)
    
    self._points = points = np.clip(points, -0.5, 1.5)
    distance = np.cumsum( np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=-1 )) )
    distance = np.insert(distance, 0, 0)
    
    speed = np.random.uniform(0.15, 1.0, size=1)[0]
    T = distance[-1] / speed
    self._maxT = np.clip(T, N * 3, N * 10)
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
    pos = np.clip(pos, 0.0, 1.0)
    self._pos = self._target.on_tick(deltaT, pos)
    return
  
  def on_render(self, window):
    super().on_render(window)
    self._target.render()
    return
  pass
###############################
class CCornerMode(CAppMode):
  def __init__(self, app):
    super().__init__(app)
    self._pos = np.zeros((2, )) + 0.5
    self._target = CSpinningTarget(app)
    self._T = 0.0
    self._radius = 0.05
    self._CORNERS = np.array([
      [0.0, 0.0],
      [0.0, 1.0],
      [1.0, 0.0],
      [1.0, 1.0],
    ], np.float32)
    self._cornerId = 0
    return
  
  def on_tick(self, deltaT):
    self._T += deltaT
    T = self._T
    R = np.abs(np.sin(T * 4)) * self._radius
    pos = np.array([np.cos(T), np.sin(T)]) * R
    pos = np.clip(self._CORNERS[self._cornerId] + pos, 0.0, 1.0)
    self._pos = self._target.on_tick(deltaT, pos)
    return
  
  def on_render(self, window):
    super().on_render(window)
    self._target.render()
    return
    
  def on_event(self, event):
    super().on_event(event)
    if event.type == G.KEYDOWN:
      N = len(self._CORNERS)
      if G.K_LEFT == event.key:
        self._paused = True
        self._cornerId = (N + self._cornerId - 1) % N
        pass
      
      if G.K_RIGHT == event.key:
        self._paused = True
        self._cornerId = (N + self._cornerId + 1) % N
        pass
    return
  pass
#####################
APP_MODES = [
  CLookAtMode,
  CCornerMode,
  CSplineMode,
  CCircleMovingMode,
  CGameMode,
]