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
from App.Utils import Colors, cv2ImageToSurface
import App.AppModes as AppModes
  
class App:
  def __init__(self, tracker, dataset, predictor):
    self._running = True
    
    self._lastPrediction = None
    self._smoothedPrediction = (0, 0)
    self._showPredictions = True
    self._showSamplesDistribution = False
    
    self._tracker = tracker
    self._eyes = [None, None]
    self._dataset = dataset
    self._predictor = predictor
    
    self._currentModeId = 0
    self._currentMode = AppModes.APP_MODES[0](self)
    return
  
  @property
  def _display_surf(self):
    return pygame.display.get_surface()
  
  @property
  def WH(self):
    return np.array(pygame.display.get_surface().get_size(), np.float32)
  
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

    self._currentMode.on_event(event)
    if event.type == G.KEYDOWN:
      if G.K_ESCAPE == event.key:
        self._running = False
        return
      
      if G.K_SPACE == event.key:
        self._running = False
        return
     
      if G.K_s == event.key:
        self._showPredictions = not self._showPredictions
        return
      
      if G.K_d == event.key:
        self._showSamplesDistribution = not self._showSamplesDistribution
        return
      
      if G.K_1 <= event.key < (G.K_1 + len(AppModes.APP_MODES)):
        self._currentModeId = ind = event.key - G.K_1
        self._currentMode = AppModes.APP_MODES[ind](self)
    return
   
  def on_tick(self, deltaT):
    lastTracked = None
    tracked = self._tracker.track()
    if not(tracked is None):
      # self._eyes[0] = cv2ImageToSurface(tracked['left eye']) if tracked['left eye visible'] else None
      # self._eyes[1] = cv2ImageToSurface(tracked['right eye']) if tracked['right eye visible'] else None
      self._currentMode.accept(tracked)
      
      lastTracked = {
        'tracked': tracked, 
        'time': time.time(),
        'pos': np.array(self._smoothedPrediction, np.float32)
      }
      pass
    #####################
    prediction = self._predictor(lastTracked)
    if not(prediction is None):
      self._lastPrediction = prediction
      predPos = self._lastPrediction[0]['coords']
      d = 128.0/4
      self._lastPrediction[0]['coords'] = np.floor(predPos * d + .5) / d
    
    if self._lastPrediction:
      factor = 0.95
      predPos = self._lastPrediction[0]['coords'][-1]
      print(predPos)
      self._smoothedPrediction = np.clip(
        np.multiply(self._smoothedPrediction, factor) + np.multiply(predPos, 1.0 - factor),
        0.0, 1.0
      )
    #####################
    self._currentMode.on_tick(deltaT)
    return
    
  def on_render(self):
    window = self._display_surf
    window.fill(Colors.SILVER)

    if self._showSamplesDistribution:
      window.blit(self._distrMap, self._distrMap.get_rect(topleft=(0, 0)))
        
    self._currentMode.on_render(window)
    self._renderEyes()
    self._renderPredictions()
    
    self.drawText('Samples: %d' % (self._dataset.totalSamples, ), (5, 95), Colors.RED)
    pygame.display.flip()
    return

  def _renderPredictions(self):
    window = self._display_surf
    wh = np.array(window.get_size())
    if not(self._lastPrediction is None):
      predicted, data, info = self._lastPrediction
      positions = predicted['coords']
      positions = np.array(positions) * wh[None]
      positions = positions.astype(np.int32)
      
      if self._showPredictions:
        for prevP, nextP in zip(positions[:-1], positions[1:]):
          pygame.draw.line(window, Colors.WHITE, prevP, nextP, 2)
          self.drawObject(tuple(nextP), R=3, color=Colors.PURPLE)
          continue
        self.drawObject(tuple(positions[-1]), R=5, color=Colors.RED)

        sp = np.multiply(self._smoothedPrediction, wh)
        self.drawObject(tuple(int(x) for x in sp), R=5, color=Colors.BLACK)
          
        self.drawText(str(positions), (5, 5), Colors.BLACK)
        pass
      self.drawText(str(info), (5, 35), Colors.BLACK)
      pass
    return

  def _renderEyes(self):
    window = self._display_surf
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

  def drawText(self, text, pos, color):
    self._display_surf.blit(
      self._font.render(text, False, color),
      pos
    )
    return

  def drawObject(self, pos, R=10, color=Colors.WHITE):
    pygame.draw.circle(self._display_surf, color, pos, R, 0)
    return

  def sampleNextGoal(self, prev=None):
    def dist(a, b): return np.sqrt(np.square(np.subtract(a, b)).sum())
    wh = np.array(self._display_surf.get_size())
    pos = (0.5, 0.5) if prev is None else np.array(prev, np.float32)
    distr, dMap = self._dataset.distribution()
    ##########
    dMap = dMap.astype(np.float32)
    dMap = np.clip(dMap, 0, dMap[dMap < dMap.mean()].mean())
    dMap /= 1 + dMap.max()
    dMap = cv2ImageToSurface(
      cv2.cvtColor(127 + (dMap * 255.0 / 2.), cv2.COLOR_GRAY2BGR).astype(np.uint8)
    )
    self._distrMap = pygame.transform.scale(dMap, tuple(wh))
    ##########
    distr = [(c, N, dist(pos, c)) for c, N in distr]
    distr = [(c, N, d) for c, N, d in distr if (0.1 < d) and (d < 0.75)]
    maxN = float(max([x[1] for x in distr]))
    if maxN < 1: maxN = 1
    distr = [(c, (maxN - N) / maxN) for c, N, d in distr if (0.1 < d) and (d < 0.75)]
    random.shuffle(distr) # shuffle same values order
    candidates = list(sorted(distr, key=lambda x: x[1]))[-16:]
    goal, _ = random.choice(candidates)
    return goal
    
def main():
  folder = os.path.dirname(__file__)
  model = CFakeModel(
    # autoregressive
    model='simple', depth=15*2*2,
    F2LArgs={'steps': 5},
    weights={'folder': folder},
    trainable=not False
  )
   
  with CThreadedEyeTracker() as tracker, CDataset(os.path.join(folder, 'Dataset'), model.timesteps) as dataset:
    with CLearnablePredictor(dataset, model=model) as predictor:
      app = App(tracker, dataset, predictor=predictor.async_infer)
      app.run()
  return
  with CThreadedEyeTracker() as tracker, CDataset(os.path.join(folder, 'Dataset'), None) as dataset:
    app = App(tracker, dataset, predictor=lambda x: None)
    app.run()
  return

if __name__ == '__main__':
  main()
