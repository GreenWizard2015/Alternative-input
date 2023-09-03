#!/usr/bin/env python
# -*- coding: utf-8 -*-.
'''
  Very simple app to control the mouse with the eyes and mouth (only for Windows)
  Mainly for testing purposes, because it is highly resource intensive (CPU/GPU go brrrrr, dah).

  Before running this app, you need to train a model.
  Also, you need to run "pip install pywin32 pywinauto keyboard" to install the dependencies.
  After that, you have to perform the steps described at https://github.com/mhammond/pywin32#installing-globally
  Sadly, but running this particular script is very tricky due to the fact that it uses pywinauto/pywin32.
'''
import numpy as np
from Core.CThreadedEyeTracker import CThreadedEyeTracker
from Core.CLearnablePredictor import CLearnablePredictor
from Core.CModelWrapper import CModelWrapper
import os
import time
import keyboard
import argparse
import threading
import pywintypes # should be imported before win32api due to a bug in pywin32
import pywinauto
from pywinauto import win32functions, win32defines

class App:
  def __init__(self, tracker, predictor, smoothingFactor, fps, lipsMinDistance):
    self._smoothingFactor = smoothingFactor
    self._fps = fps
    self._lipsMinDistance = lipsMinDistance
    
    self._lastPrediction = None
    self._smoothedPrediction = (0, 0)
    self._tracker = tracker
    self._predictor = predictor

    self._mouthOpen = False
    self._mouthWasOpen = False

    self._done = threading.Event()
    return

  def on_keypress(self, event):
    if 'esc' == event.name:
      self._done.set()
    return
  
  def on_tick(self):
    lastTracked = None
    tracked = self._tracker.track()
    if not(tracked is None):
      self._mouthOpen = self._lipsMinDistance <= tracked['lips distance']
      T = time.time()
      lastTracked = {
        'tracked': {**tracked, 'time': T},
        'time': T,
        'pos': np.array(self._smoothedPrediction, np.float32)
      }
      prediction = self._predictor(lastTracked['tracked'])
      if not(prediction is None):
        self._lastPrediction = prediction
      pass
    #####################
    
    if self._lastPrediction:
      factor = self._smoothingFactor
      pred = self._lastPrediction[0]
      predPos = pred['coords']
      self._smoothedPrediction = np.clip(
        np.multiply(self._smoothedPrediction, factor) + np.multiply(predPos, 1.0 - factor),
        0.0, 1.0
      )
    #####################
    pos = np.multiply(self._smoothedPrediction, self._screen)
    pos = tuple(pos.astype(np.int32))
    try:
      if self._mouthOpen and not self._mouthWasOpen:
        pywinauto.mouse.click(button='left', coords=pos)
      else:
        pywinauto.mouse.move(pos)
        pass
    except:
      pass
    self._mouthWasOpen = self._mouthOpen
    return

  def run(self):
    keyboard.on_press(self.on_keypress, suppress=False)
    try:
      pywinauto.timings.Timings.after_setcursorpos_wait = 0.0
      pywinauto.timings.Timings.after_clickinput_wait = 0.0
      
      self._screen = np.array([
        win32functions.GetSystemMetrics(win32defines.SM_CXSCREEN),
        win32functions.GetSystemMetrics(win32defines.SM_CYSCREEN)
      ])
      while not self._done.wait(1.0 / self._fps):
        self.on_tick()
        continue
    finally:
      keyboard.unhook_all()
    return
    
def main(args):
  folder = os.path.dirname(__file__)
  folder = os.path.join(folder, 'Data')
  model = CModelWrapper(timesteps=5, weights=dict(folder=folder, postfix=args.model))
   
  with CThreadedEyeTracker(fps=args.fps) as tracker:
    with CLearnablePredictor(model=model, fps=args.fps) as predictor:
      app = App(
        tracker, predictor=predictor.async_infer, 
        smoothingFactor=args.smoothing,
        fps=args.fps,
        lipsMinDistance=args.lips_distance
      )
      app.run()
  return

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--smoothing', type=float, default=0.95)
  parser.add_argument('--fps', type=int, default=10)
  parser.add_argument('--lips-distance', type=float, default=50)
  parser.add_argument('--model', type=str, default='best')
  main(parser.parse_args())
 