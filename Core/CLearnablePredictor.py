import threading
from .CFakeModel import CFakeModel
import Utils

class CLearnablePredictor:
  def __init__(self, dataset):
    self._dataset = dataset
    self._lock = threading.Lock()
    self._done = threading.Event()
    self._inferData = None
    self._inferResults = None
    self._model = CFakeModel()
    return

  def __enter__(self):
    self._thread = threading.Thread(target=self._trainLoop)
    self._thread.start()
    return self

  def __exit__(self, type, value, traceback):
    self._done.set()
    self._thread.join()
    return

  def async_infer(self, data):
    with self._lock:
      self._inferData = data
      res = self._inferResults
      self._inferResults = None
    return res
  
  def _trainLoop(self):
    while not self._done.isSet():
      self._train()
      self._infer()
      continue
    return
  
  def _train(self):
    data = self._dataset.sample()
    if data:
      info = self._model.fit(data)
    return
  
  def _infer(self):
    with self._lock:
      data = self._inferData
      self._inferData = None
      
    if data is None: return
    
    sample = Utils.tracked2sample(data['tracked'])
    res = self._model(Utils.samples2inputs([sample]))[0]
    with self._lock:
      self._inferResults = (res, data, {'samples': len(self._dataset)})
    return
  