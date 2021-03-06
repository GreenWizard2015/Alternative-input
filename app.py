#!/usr/bin/env python
# -*- coding: utf-8 -*-.
import numpy as np
import pygame
import pygame.locals as G
from Core.CEyeTracker import CEyeTracker
from Core.CThreadedEyeTracker import CThreadedEyeTracker
from Core.CDataset import CDataset
from Core.CLearnablePredictor import CLearnablePredictor
import cv2
import os
from Core.CFakeModel import CFakeModel
import random
import time
  
def normalized(a, axis=-1, order=2):
  l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
  l2[l2==0] = 1
  return a / np.expand_dims(l2, axis)

def cv2ImageToSurface(cv2Image):
    if cv2Image.dtype.name == 'uint16':
        cv2Image = (cv2Image / 256).astype('uint8')
    size = cv2Image.shape[1::-1]
    if len(cv2Image.shape) == 2:
        cv2Image = np.repeat(cv2Image.reshape(size[1], size[0], 1), 3, axis = 2)
        format = 'RGB'
    else:
        format = 'RGBA' if cv2Image.shape[2] == 4 else 'RGB'
        cv2Image[:, :, [0, 2]] = cv2Image[:, :, [2, 0]]
    surface = pygame.image.frombuffer(cv2Image.flatten(), size, format)
    return surface.convert_alpha() if format == 'RGBA' else surface.convert()
  
class Colors:
  BLACK = (0, 0, 0)
  SILVER = (192, 192, 192)
  WHITE = (255, 255, 255)
  BLUE = (0, 0, 255)
  GREEN = (0, 255, 0)
  RED = (255, 0, 0)
  PURPLE = (255, 0, 255)

class App:
  def __init__(self, tracker, dataset, predictor):
    self._running = True
    self._paused = True # False
    self._speed = 55 * 2 * 2
    self._pos = (25, 25)
    self._goal = self._pos
    self._lastTracked = None
    self._lastPrediction = None
    self._smoothedPrediction = (0, 0)
    self._errorHistory = [0.0]
    self._showSamplesDistribution = False
    self._showPredictions = True
    
    self._tracker = tracker
    self._eyes = [None, None]
    self._dataset = dataset
    self._predictor = predictor
    return
  
  @property
  def _display_surf(self):
    return pygame.display.get_surface()
  
  def on_init(self):
    pygame.init()
    
    info = pygame.display.Info()
    w = info.current_w
    h = info.current_h
    pygame.display.set_mode((w, h), pygame.FULLSCREEN)
    
    pygame.display.set_caption('App')
    self._font = pygame.font.Font(pygame.font.get_default_font(), 16)
    self._running = True
    return True
  
  def on_event(self, event):
    if event.type == G.QUIT:
      self._running = False
      return

    if event.type == G.KEYDOWN:
      if G.K_ESCAPE == event.key:
        self._running = False
        return
      
      if G.K_SPACE == event.key:
        self._running = False
        return
      
      if G.K_p == event.key:
        self._paused = not self._paused
        return
      
      if G.K_d == event.key:
        self._showSamplesDistribution = not self._showSamplesDistribution
        return
      
      if G.K_s == event.key:
        self._showPredictions = not self._showPredictions
        return
    return
   
  def on_tick(self, deltaT):
    wh = np.array(self._display_surf.get_size(), np.float32)
    tracked = self._tracker.track()
    if not(tracked is None):
      # self._eyes[0] = cv2ImageToSurface(tracked['left eye']) if tracked['left eye visible'] else None
      # self._eyes[1] = cv2ImageToSurface(tracked['right eye']) if tracked['right eye visible'] else None
      if  not self._paused:
        self._dataset.store(tracked, np.array(self._pos) / wh, time.time())
      
      self._lastTracked = {
        'tracked': tracked, 
        'time': time.time(),
        'debugGoal': np.array(self._pos) / wh,
        'pos': np.array(self._smoothedPrediction, np.float32)
      }
      pass
    #####################
    if not(self._lastTracked is None):
      prediction = self._predictor(self._lastTracked)
      if prediction:
        self._lastPrediction = prediction
        self._lastTracked = None
        
        predPos = prediction[0]['coords'][-1]
        diff = np.subtract(predPos, prediction[1]['debugGoal'])
        self._errorHistory.append(
          np.sqrt(np.square(diff).sum())
        )
        self._errorHistory = self._errorHistory[-25:]
      pass
    
    if self._lastPrediction:
      factor = 0.95
      predPos = self._lastPrediction[0]['coords'][-1]
      self._smoothedPrediction = np.clip(
        np.multiply(self._smoothedPrediction, factor) + np.multiply(predPos, 1.0 - factor),
        0.0, 1.0
      )
      
    #####################
    vec = normalized(np.subtract(self._goal, self._pos))[0]
    self._pos = np.add(self._pos, vec * self._speed * deltaT)
    
    dist = np.sqrt(np.square(np.subtract(self._pos, self._goal)).sum())
    if dist < 3.0:
      self._nextGoal()
    return
    
  def on_render(self):
    window = self._display_surf
    wh = np.array(window.get_size())
    window.fill(Colors.SILVER)

    if self._showSamplesDistribution:
      window.blit(self._distrMap, self._distrMap.get_rect(topleft=(0, 0)))
      pass
    
    if self._paused:
      self._drawText('Paused', (55, 55), Colors.RED)

    self._drawObject(tuple(int(x) for x in self._pos))
    
    self._drawText(
      '%.5f' % (np.mean(self._errorHistory)),
      tuple(int(x) for x in self._pos),
      Colors.RED
    )

    if not (self._eyes[0] is None):
      x = pygame.transform.scale(self._eyes[0], (128, 128))
      window.blit(
        x,
        x.get_rect(topleft=window.get_rect().inflate(-10, -10).topleft)
      )
    
    if not (self._eyes[1] is None):
      x = pygame.transform.scale(self._eyes[1], (128, 128))
      window.blit(
        x,
        x.get_rect(topleft=window.get_rect().inflate(-10, -10 - 2*128).topleft)
      )

    if self._showPredictions and not(self._lastPrediction is None):
      predicted, data, info = self._lastPrediction
      positions = predicted['coords']
      
      positions = np.array(positions) * wh[None]
      positions = positions.astype(np.int32)
      for prevP, nextP in zip(positions[:-1], positions[1:]):
        pygame.draw.line(window, Colors.WHITE, prevP, nextP, 2)
        self._drawObject(tuple(nextP), R=3, C=Colors.PURPLE)
        continue
      self._drawObject(tuple(positions[-1]), R=5, C=Colors.RED)
      self._drawText(str(positions), (5, 5), Colors.BLACK)
      self._drawText(str(info), (int(self._pos[0]) - 95, int(self._pos[1]) + 15), Colors.BLACK)
      
      sp = np.multiply(self._smoothedPrediction, wh)
      self._drawObject(tuple(int(x) for x in sp), R=5, C=Colors.BLACK)
      pass
    pygame.display.flip()
    return
 
  def run(self):
    if not self.on_init():
      self._running = False
      
    T = pygame.time.get_ticks()
    clock = pygame.time.Clock()
    while self._running:
      for event in pygame.event.get():
        self.on_event(event)

      self.on_tick((pygame.time.get_ticks() - T) / 1000)
      self.on_render()
      T = pygame.time.get_ticks()
      clock.tick(30)
      continue
      
    pygame.quit()
    return

  def _drawText(self, text, pos, color):
    self._display_surf.blit(
      self._font.render(text, False, color),
      pos
    )
    return

  def _drawObject(self, pos, R=10, C=Colors.WHITE):
    pygame.draw.circle(self._display_surf, C, pos, R, 0)
    return
  
  def _nextGoal(self):
    def dist(a, b): return np.square(np.subtract(a, b)).sum()
    wh = np.array(self._display_surf.get_size())
    pos = np.array(self._pos, np.float) / wh
    distr, dMap = self._dataset.distribution()
    ##########
    dMap = dMap.astype(np.float32)
    dMap /= 1 + dMap.max()
    dMap = cv2ImageToSurface(
      cv2.cvtColor(127 + (dMap * 255.0 / 2.), cv2.COLOR_GRAY2BGR).astype(np.uint8)
    )
    self._distrMap = pygame.transform.scale(dMap, self._display_surf.get_size())
    ##########
    distr = [(c, N, dist(pos, c)) for c, N in distr]
    distr = [(c, N, d) for c, N, d in distr if (0.1 < d) and (d < 0.75)]
    maxN = float(max([x[1] for x in distr]))
    if maxN < 1: maxN = 1
    distr = [(c, (maxN - N) / maxN) for c, N, d in distr if (0.1 < d) and (d < 0.75)]
    random.shuffle(distr) # shuffle same values order
    candidates = list(sorted(distr, key=lambda x: x[1]))[-16:]
    goal, _ = random.choice(candidates)
    self._goal = np.multiply(goal, wh)
    return
  
def main():
  folder = os.path.dirname(__file__)
  with CThreadedEyeTracker() as tracker, CDataset(os.path.join(folder, 'Dataset')) as dataset:
    model = CFakeModel('autoregressive', depth=5, weights=os.path.join(folder, 'autoregressive.h5'), trainable=not True)
#     model = CFakeModel('autoregressive', depth=5)
#     model = CFakeModel('simple', weights=os.path.join(folder, 'simple.h5'), trainable=True)
    with CLearnablePredictor(dataset, model=model) as predictor:
      app = App(tracker, dataset, predictor=predictor.async_infer)
      app.run()
  pass

if __name__ == '__main__':
  main()
