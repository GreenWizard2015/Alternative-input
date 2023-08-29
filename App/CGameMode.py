import numpy as np
import pygame
import pygame.locals as G
from App.Utils import Colors
from App.CAppMode import CAppMode

def _adjustPos(x, power=4):
  if x < 0.5:
    x = 2 * x # scale to 0..1
    x = np.power(x, power) # make it more often on edges
    return x / 2 # scale back to 0..0.5
  
  return 1.0 - _adjustPos(1.0 - x)

class CGameMode(CAppMode):
  def __init__(self, app):
    super().__init__(app)
    self._pos = np.zeros((2, )) + 0.5
    self._T = 0.0
    self._currentRadius = 0.0
    self._radiusPerSecond = 0.01
    self._hits = 0
    self._maxHits = 5

    self._inRangeHits = 0
    self._inRangeMaxHits = 5

    self._totalTime = 0
    self._totalHits = 0
    self._totalDistance = 0

    self._totalObservations = 0
    self._totalObservationsDistance = 0
    self._probPower = 1
    return

  def on_sample(self, tracked):
    if not self._app.hasPredictions: return
    if self._paused: return
    if 0 < self._hits:
      self._app._dataset.store(tracked, np.array(self._pos))
    return
    
  def on_tick(self, deltaT):
    self._T += deltaT
    T = self._T
    self._currentRadius = T * self._radiusPerSecond
    return
  
  def on_render(self, window):
    if not self._app.hasPredictions:
      self._app.drawText(
        'Game mode requires predictions to be provided by the model',
        pos=(window.get_width() // 2, window.get_height() // 3),
        color=Colors.RED,
        center=True, scale=4.0
      )
    
    wh = np.array(window.get_size())
    pos = tuple(np.multiply(wh, self._pos).astype(np.int32))
    self._app.drawObject(pos, color=Colors.WHITE) # draw the target for focusing on
    # second circle
    R = np.multiply(wh, self._currentRadius).min().astype(np.int32)
    clr = Colors.RED if self._hits == 0 else Colors.GREEN
    pygame.draw.circle(window, clr, pos, int(R), width=1)

    # score at the top center
    hits = self._totalHits
    if 0 < hits:
      self._app.drawText(
        'Hits: %d, accuracy: %.4f, time: %.2f, obs. dist.: %.4f' % (
          hits,self._totalDistance / hits, self._totalTime / hits,
          self._totalObservationsDistance / self._totalObservations if 0 < self._totalObservations else 0.0
        ),
        pos=(wh[0] // 2, 80),
        color=Colors.BLACK,
        center=True,
      )
    
    self._app.drawText(
      'Power: %d, Hits: %d / %d, In range: %d / %d' % (
        self._probPower,
        self._hits, self._maxHits,
        self._inRangeHits, self._inRangeMaxHits
      ),
      color=Colors.BLACK,
      pos=(wh[0] // 2, 80 + 30),
      scale=0.75, center=True,
    )
    return
  
  def on_prediction(self, pos, tracked):
    if not self._app.hasPredictions: return
    pos = np.array(pos).reshape((2, ))
    # check if the click is inside the circle
    D = np.square(np.subtract(pos, self._pos)).sum()
    D = np.sqrt(D)
    # calculate the global accuracy
    if 0 < self._hits:
      self._totalObservations += 1
      self._totalObservationsDistance += D

    if D < self._currentRadius:
      self._inRangeHits += 1
      if self._inRangeMaxHits <= self._inRangeHits:
        self._inRangeHits = 0
        self._hit(D)
        pass
    else:
      self._inRangeHits = 0
    return
  
  def _hit(self, D):
    if 0 < self._hits: # Only if active
      self._totalHits += 1
      self._totalDistance += D
      self._totalTime += self._T
    
    self._T = 0.0
    self._currentRadius = 0.0

    self._hits += 1
    if self._maxHits <= self._hits:
      self._nextGoal()
    return
  
  def _nextGoal(self):
    pos = np.random.random((2, ))
    if 1 < self._probPower:
      pos = np.array([_adjustPos(x, self._probPower) for x in pos])
      
    self._pos = np.clip(pos, 0.0, 1.0)
    self._hits = 0
    self._inRangeHits = 0
    return
  
  def on_event(self, event):
    if G.KEYDOWN == event.type:
      if G.K_UP == event.key: self._probPower += 1
      if G.K_DOWN == event.key: self._probPower -= 1
      self._probPower = np.clip(self._probPower, 1, 10)

      # numpad plus and minus
      if G.K_KP_PLUS == event.key: self._inRangeMaxHits += 1
      if G.K_KP_MINUS == event.key: self._inRangeMaxHits -= 1
      self._inRangeMaxHits = np.clip(self._inRangeMaxHits, 1, 25)
      pass
    return super().on_event(event)
  pass
