#!/usr/bin/env python
# -*- coding: utf-8 -*-.
import numpy as np
import pygame
import pygame.locals as G
from Core.CThreadedEyeTracker import CThreadedEyeTracker
from Core.CDataset import CDataset
from Core.CLearnablePredictor import CLearnablePredictor
from Core.CDemoModel import CDemoModel
import os, time
from App.Utils import Colors
import App.AppModes as AppModes
from App.CRandomIllumination import CRandomIllumination

class App:
  def __init__(self, tracker, dataset, predictor):
    self._running = True
    
    self._lastPrediction = None
    self._smoothedPrediction = (0, 0)
    self._showPredictions = True
    
    self._tracker = tracker
    self._dataset = dataset
    self._predictor = predictor
    
    self._currentModeId = 0
    self._currentMode = AppModes.APP_MODES[0](self)
    
    self._history = []
    self._illumination = CRandomIllumination()
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
      
      if G.K_1 <= event.key < (G.K_1 + len(AppModes.APP_MODES)):
        self._currentModeId = ind = event.key - G.K_1
        self._currentMode = AppModes.APP_MODES[ind](self)
    return
   
  def on_tick(self, deltaT):
    lastTracked = None
    tracked = self._tracker.track()
    if not(tracked is None):
      self._currentMode.accept(tracked)
      
      lastTracked = {
        'tracked': tracked,
        'pos': np.array(self._smoothedPrediction, np.float32)
      }
      pass
    #####################
    prediction = self._predictor(lastTracked)
    if not(prediction is None):
      self._lastPrediction = prediction
      pred = prediction[0]

      self._history.append(pred['coords'])
      self._history = self._history[-15:]
      pass
    #####################
    if self._lastPrediction:
      factor = 0.9
      pred = self._lastPrediction[0]
      predPos = pred['coords']
      self._smoothedPrediction = np.clip(
        np.multiply(self._smoothedPrediction, factor) + np.multiply(predPos, 1.0 - factor),
        0.0, 1.0
      )
    #####################
    self._currentMode.on_tick(deltaT)
    self._illumination.on_tick(deltaT)
    return
    
  def on_render(self):
    window = self._display_surf
    # take color from Colors.asList based on current time, change every 5 seconds
    clr = Colors.asList[int(time.time() / 5) % len(Colors.asList)]
    window.fill(clr)
    
    self._illumination.on_render(window)
    self._currentMode.on_render(window)
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
      positions = self._history
      positions = np.array(positions) * wh[None]
      positions = positions.astype(np.int32)
      
      if self._showPredictions:
        for prevP, nextP in zip(positions[:-1], positions[1:]):
          pygame.draw.line(window, Colors.WHITE, prevP, nextP, 2)
          self.drawObject(tuple(nextP), R=3, color=Colors.PURPLE)
          continue
        self.drawObject(tuple(positions[-1]), R=5, color=Colors.RED)

        sp = np.multiply(self._smoothedPrediction, wh).astype(np.int32)
        self.drawObject(tuple(sp), R=5, color=Colors.BLACK)

        self.drawText(str(positions), (5, 5), Colors.BLACK)
        pass
      self.drawText(str(info), (5, 35), Colors.BLACK)
      pass
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
    # if white - draw circle with black border
    if np.all(np.equal(color, Colors.WHITE)):
      pygame.draw.circle(self._display_surf, Colors.BLACK, pos, R + 4, 0)
      pygame.draw.circle(self._display_surf, Colors.RED, pos, R + 2, 0)
      
    pygame.draw.circle(self._display_surf, color, pos, R, 0)
    return

def modeA(folder):
  model = CDemoModel(trainable=not True, timesteps=5, weights=dict(folder=folder, postfix='latest'))
  with CThreadedEyeTracker() as tracker, CDataset(os.path.join(folder, 'Dataset'), model.timesteps) as dataset:
    with CLearnablePredictor(dataset, model=model) as predictor:
      app = App(tracker, dataset, predictor=predictor.async_infer)
      app.run()
      model.save(folder)
      pass
    pass
  return

def modeB(folder, datasetName='Dataset-test'):
  with CThreadedEyeTracker() as tracker, CDataset(os.path.join(folder, datasetName), 5) as dataset:
    app = App(tracker, dataset, predictor=lambda *x: None)
    app.run()
    pass
  return

def main():
  folder = os.path.join(os.path.dirname(__file__), 'Data')
  modeA(folder)
  # modeB(folder, datasetName='Dataset')
  # modeB(folder, datasetName='Dataset-test')
  return

if __name__ == '__main__':
  main()
