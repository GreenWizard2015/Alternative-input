#!/usr/bin/env python
# -*- coding: utf-8 -*-.
import numpy as np
import pygame
import pygame.locals as G
import cv2
from Core.CThreadedEyeTracker import CThreadedEyeTracker
from Core.CDataset import CDataset
from Core.CLearnablePredictor import CLearnablePredictor
from Core.CDummyPredictor import CDummyPredictor
from Core.CModelWrapper import CModelWrapper
from Core.Utils import FACE_MESH_INVALID_VALUE
import os, time
from App.Utils import Colors
import App.AppModes as AppModes
from App.CRandomIllumination import CRandomIllumination
from App.CBackground import CBackground
import argparse

class App:
  def __init__(
    self, tracker, dataset, predictor, 
    fps=30, showWebcam=False, hasPredictions=True, showFaceMesh=False
  ):
    self._showFaceMesh = showFaceMesh
    self._faceMesh = None
    self._canPredict = hasPredictions
    self._fps = fps
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
    self._enableIllumination = False
    self._illumination = CRandomIllumination()
    self._background = CBackground()

    self._cameraView = None
    self._cameraSurface = None

    if showWebcam:
      self._cameraView = np.array([(50, 200), (50 + 300, 200 + 300)])
      self._cameraSurface = pygame.Surface(self._cameraView[1] - self._cameraView[0])

    self._predictorMaskFace = False
    self._predictorMaskLeftEye = False
    self._predictorMaskRightEye = False
    return
  
  @property
  def hasPredictions(self): return self._canPredict
  
  def _transformTracked(self, tracked):
    if tracked is None: return None

    tracked = tracked['tracked']
    res = dict(**tracked)

    if self._predictorMaskFace:
      res['face points'] = np.full_like(tracked['face points'], FACE_MESH_INVALID_VALUE)

    if self._predictorMaskLeftEye:
      res['left eye'] = np.full_like(tracked['left eye'], 0.0)

    if self._predictorMaskRightEye:
      res['right eye'] = np.full_like(tracked['right eye'], 0.0)
    return res
  
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

    self._background.on_event(event)
    self._currentMode.on_event(event)
    if event.type == G.KEYDOWN:
      if G.K_ESCAPE == event.key:
        self._running = False
        return
      
      if G.K_s == event.key:
        self._showPredictions = not self._showPredictions
        return
      
      if G.K_1 <= event.key < (G.K_1 + len(AppModes.APP_MODES)):
        self._currentModeId = ind = event.key - G.K_1
        self._currentMode = AppModes.APP_MODES[ind](self)
      # toggle illumination (L)
      if G.K_l == event.key:
        self._enableIllumination = not self._enableIllumination
      # predictor masks switches (F1, F2, F3)
      if G.K_F1 == event.key:
        self._predictorMaskFace = not self._predictorMaskFace

      if G.K_F2 == event.key:
        self._predictorMaskLeftEye = not self._predictorMaskLeftEye

      if G.K_F3 == event.key:
        self._predictorMaskRightEye = not self._predictorMaskRightEye
    return
   
  def on_tick(self, deltaT):
    lastTracked = None
    tracked = self._tracker.track()
    if not(tracked is None):
      self._currentMode.on_sample(tracked)
      
      if not(self._cameraView is None):
        WH = self._cameraView[1] - self._cameraView[0]
        raw = tracked['raw']
        raw = cv2.resize(raw, tuple(WH.astype(np.int32)))
        surf = pygame.surfarray.pixels3d(self._cameraSurface)
        raw = raw[:, :, ::-1] # RGB -> BGR
        raw = np.swapaxes(raw, 0, 1) # H x W x C -> W x H x C
        surf[:, :, :] = raw # BGR -> RGB
        del surf # release surface
        pass

      if self._showFaceMesh:
        self._faceMesh = tracked['face points'].copy()
        
      lastTracked = {
        'tracked': tracked,
        'pos': np.array(self._smoothedPrediction, np.float32)
      }
      pass
    #####################
    prediction = self._predictor( self._transformTracked(lastTracked) )
    if not(prediction is None):
      self._lastPrediction = prediction
      pred = prediction[0]
      predPos = pred['coords']

      self._history.append(predPos)
      self._history = self._history[-15:]
      self._currentMode.on_prediction(predPos, lastTracked)
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
    self._background.on_tick(deltaT)
    self._currentMode.on_tick(deltaT)
    if self._enableIllumination:
      self._illumination.on_tick(deltaT)
    return
    
  def on_render(self, fps=0.0):
    window = self._display_surf
    self._background.on_render(window)
    
    if self._enableIllumination:
      self._illumination.on_render(window)

    if not(self._cameraSurface is None): # render camera surface
      window.blit(self._cameraSurface, self._cameraView)
    
    self._currentMode.on_render(window)
    if self._currentMode.paused:
      wh = np.array(window.get_size())
      txt = 'Collection paused'
      self.drawText(txt, wh // 2, Colors.RED, scale=2.0, center=True)
      
    self._renderPredictions()
    
    self._renderInfo(fps=fps)
    pygame.display.flip()
    return
  
  def _renderInfo(self, fps):
    self.drawText('Samples: %d' % (self._dataset.totalSamples, ), (5, 95), Colors.RED)
    modes = []
    if self._predictorMaskFace: modes.append('no face')
    if self._predictorMaskLeftEye: modes.append('no left eye')
    if self._predictorMaskRightEye: modes.append('no right eye')

    if 0 < len(modes):
      self.drawText('%s' % (', '.join(modes), ), (5, 95 + 25), Colors.GREEN)
    
    self.drawText('FPS: %.1f' % (fps, ), (5, 95 + 25 + 25), Colors.BLACK)

    if self._faceMesh is not None:
      scaled = np.multiply(self._faceMesh, self.WH[None])
      scaled = scaled.astype(np.int32)
      for p in scaled:
        pygame.draw.circle(self._display_surf, Colors.RED, tuple(p), 2, 0)
        continue
      pass
    return

  def _renderPredictions(self):
    window = self._display_surf
    wh = np.array(window.get_size())
    if self._showPredictions and (0 < len(self._history)):
      positions = np.array(self._history) * wh[None]
      positions = positions.astype(np.int32)
      for prevP, nextP in zip(positions[:-1], positions[1:]):
        pygame.draw.line(window, Colors.WHITE, prevP, nextP, 2)
        self.drawObject(tuple(nextP), R=3, color=Colors.PURPLE)
        continue
      self.drawObject(tuple(positions[-1]), R=5, color=Colors.RED)

      sp = np.multiply(self._smoothedPrediction, wh).astype(np.int32)
      self.drawObject(tuple(sp), R=5, color=Colors.BLACK)

      self.drawText(str(positions), (5, 5), Colors.BLACK)
      pass

    if not(self._lastPrediction is None):
      predicted, data, info = self._lastPrediction
      # self.drawText(str(info), (5, 35), Colors.BLACK)
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

      TMs = (pygame.time.get_ticks() - T) / 1000.0
      fps = 1.0 / TMs if 0.0 < TMs else 0.0
      self.on_tick(TMs)
      self.on_render(fps=fps)
      T = pygame.time.get_ticks()
      clock.tick(self._fps)
      continue
      
    pygame.quit()
    return

  def drawText(self, text, pos, color, scale=1.0, center=False):
    textSurface = self._font.render(text, False, color)
    if 1.0 != scale:
      textSurface = pygame.transform.scale(
        textSurface, 
        (int(textSurface.get_width() * scale), int(textSurface.get_height() * scale))
      )

    if center:
      pos = np.subtract(pos, np.divide(textSurface.get_size(), 2))
    
    pos = tuple(int(x) for x in pos)
    self._display_surf.blit(textSurface, pos)
    return

  def drawObject(self, pos, R=10, color=Colors.WHITE):
    # if white - draw target
    if np.all(np.equal(color, Colors.WHITE)): return self.drawTarget(pos, R=R)
    
    pygame.draw.circle(self._display_surf, color, pos, R, 0)
    return

  def drawTarget(self, pos, R=10):
    T = int(time.time())
    surf = self._display_surf
    colors = Colors.asList
    for i in reversed(range(2, R, 2)):
      color = colors[(i * 7 + T) % len(colors)]
      pygame.draw.circle(surf, color, pos, i, 0)
      continue
    return

def _modelFromArgs(args):
  if 'none' == args.model.lower(): return None
  return CModelWrapper(
    timesteps=args.steps, 
    weights=dict(folder=args.folder, postfix=args.model)
  )

def _predictorFromArgs(args):
  model = _modelFromArgs(args)
  if model is None: return CDummyPredictor()
  return CLearnablePredictor(model=model, fps=args.fps)

def main(args):
  folder = args.folder
  with CThreadedEyeTracker() as tracker, CDataset(os.path.join(folder, 'Dataset'), args.steps) as dataset:
    with _predictorFromArgs(args) as predictor:
      app = App(tracker, dataset, predictor=predictor.async_infer, fps=args.fps, hasPredictions=predictor.canPredict)
      app.run()
    pass
  return

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--folder', type=str, default=os.path.join(os.path.dirname(__file__), 'Data'))
  parser.add_argument('--steps', type=int, default=5)
  # if 'none' - no model will be used
  parser.add_argument('--model', type=str, default='best')
  parser.add_argument('--fps', type=int, default=30)
  main(parser.parse_args())
  pass