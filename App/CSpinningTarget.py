import numpy as np
from App.Utils import Colors, rotate

class CSpinningTarget:
  def __init__(self, app):
    super().__init__()
    self._app = app
    self._angle = np.random.uniform(high=2.0 * np.pi, size=1)[0]
    self._radius = 0.01
    self._pos = np.zeros((2, )) + 0.5
    self._T = 0.0
    self._TScale = 10
    return
  
  def on_tick(self, deltaT, wheelPos):
    self._T += deltaT
    self._pos = wheelPos

    T = (self._T / self._TScale) % (2 * np.pi)
    self._radius = 0.001 + np.cos(T) * 0.015
    self._angle = (self._angle + .1) % (2 * np.pi)
    ###############
    wh = self._app.WH
    mainPos = np.multiply(wh, self._pos)
    vec = np.multiply(wh, (self._radius, 0.0))
    return np.divide(mainPos + rotate(vec, self._angle), wh)
  
  def render(self):
    wh = self._app.WH
    mainPos = np.multiply(wh, self._pos)
    vec = np.multiply(wh, (self._radius, 0.0))
    N = 5
    angles = np.linspace(0., 2 * np.pi, num=N, endpoint=False)
    for i, angle in enumerate(angles[1:]):
      pos = mainPos + rotate(vec, self._angle + angle)
      self._app.drawObject(
        tuple(int(x) for x in pos), 
        color=Colors.PURPLE,
        R=N + 3 - i
      )
      continue
    
    pos = mainPos + rotate(vec, self._angle)
    self._app.drawObject(
      tuple(int(x) for x in pos), 
      color=Colors.WHITE,
      R=N + 4
    )
    
    clp = np.clip(pos, 0, wh)
    if not np.allclose(clp, pos):
      self._app.drawObject(tuple(int(x) for x in clp), color=Colors.PURPLE, R=3)
    return
