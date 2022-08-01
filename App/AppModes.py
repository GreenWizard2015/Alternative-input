import numpy as np
import pygame.locals as G
from App.Utils import Colors, normalized
from App.CSpinningTarget import CSpinningTarget
import time
import scipy.interpolate as sInterp

class CAppMode:
  def __init__(self, app):
    self._app = app
    self._paused = True
    return
  
  def on_event(self, event):
    if event.type == G.KEYDOWN:
      if event.key in [G.K_p, G.K_RETURN]:
        self._paused = not self._paused
    return
  
  def on_render(self, window):
    if self._paused:
      self._app.drawText('Paused', (55, 55), Colors.RED)
    return
  
  def accept(self, tracked):
    if self._paused: return
    self._app._dataset.store(tracked, np.array(self._pos), time.time())
    return
  pass

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

class CFollowMode(CMoveToGoal):
  def _nextGoal(self, old):
    return self._app.sampleNextGoal(old)

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
        self._active = True
        return
    return

  def accept(self, tracked):
    if self._active:
      super().accept(tracked)
    return
  
  def _reset(self):
    path = np.array([
      [-1,  1],
      [ 1,  1],
      [ 1, -1],
      [-1, -1],
      [-1,  1],
    ], np.float32)
    lvl = (self._maxLevel - self._level) / ((2.0 * self._maxLevel) + 0)
    self._pos, self._goal, *self._path = 0.5 + lvl * path
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
    self._goal = None
    self._visibleT = 5.0
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
    if self._goal is None: self._next()
    
    if self._active:
      dT = time.time() - self._startT
      if self._visibleT < dT: self._next()
    else:
      self._startT = time.time()
    return
  
  def _next(self):
    self._goal = self._app.sampleNextGoal(self._goal)
    self._active = False
    self._startT = None
    return
  
  def accept(self, tracked):
    if self._active:
      super().accept(tracked)
    return
  
  def on_render(self, window):
    super().on_render(window)
    wh = np.array(window.get_size())
    pos = tuple(int(x) for x in np.multiply(wh, self._goal))
    
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
    
    speed = np.random.uniform(0.15, 1.0, size=1)[0]
    T = distance[-1] / speed
    self._maxT = np.clip(T, 5, 20)
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
  CCornerMode,
  CSplineMode,
  CFollowMode,
  CCircleMovingMode,
  CLookAtMode
]