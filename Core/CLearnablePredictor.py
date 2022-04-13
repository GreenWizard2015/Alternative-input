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
      trainInfo = self._train()
      self._infer(trainInfo)
      continue
    return
  
  def _train(self):
    data = self._dataset.sample()
    info = {}
    if data:
      info = self._model.fit(data)
    return info
  
  def _infer(self, trainInfo):
    with self._lock:
      data = self._inferData
      self._inferData = None
      
    if data is None: return
    
    sample = Utils.tracked2sample(data['tracked'], dropout=0.25)
    res = self._model(Utils.samples2inputs([sample]))
    with self._lock:
      self._inferResults = (res, data, {'samples': len(self._dataset), **trainInfo})
    return
  