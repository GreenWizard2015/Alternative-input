#!/usr/bin/env python
# -*- coding: utf-8 -*-.
import numpy as np
from Core.CThreadedEyeTracker import CThreadedEyeTracker
from Core.CLearnablePredictor import CLearnablePredictor
import os
import time
import pywinauto
from pywinauto import win32functions, win32defines
import keyboard
from Core.CDemoModel import CDemoModel

class App:
  def __init__(self, tracker, predictor):
    self._running = True
    
    self._lastPrediction = None
    self._smoothedPrediction = (0, 0)
    self._tracker = tracker
    self._predictor = predictor

    self._mouthOpen = False
    self._mouthWasOpen = False
    return

  def on_keypress(self, event):
    if 'esc' == event.name:
      self._running = False
    return
  
  def on_tick(self):
    lastTracked = None
    tracked = self._tracker.track()
    if not(tracked is None):
      self._mouthOpen = 50 <= tracked['lips distance']
      T = time.time()
      lastTracked = {
        'tracked': {**tracked, 'time': T},
        'time': T,
        'pos': np.array(self._smoothedPrediction, np.float32)
      }
      pass
    #####################
    prediction = self._predictor(lastTracked)
    if not(prediction is None):
      self._lastPrediction = prediction
    
    if self._lastPrediction:
      factor = 0.97
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
      while self._running:
        self.on_tick()
        time.sleep(0.01)
        continue
    finally:
      keyboard.unhook_all()
    return
    
def main():
  folder = os.path.dirname(__file__)
  folder = os.path.join(folder, 'Data')
  model = CDemoModel(timesteps=5, weights=dict(folder=folder, postfix='latest'), trainable=False)
   
  with CThreadedEyeTracker() as tracker:
    with CLearnablePredictor(dataset=None, model=model) as predictor:
      app = App(tracker, predictor=predictor.async_infer)
      app.run()
  return

if __name__ == '__main__':
  main()
 