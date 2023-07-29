import threading
import Core.Utils as Utils
import numpy

class CLearnablePredictor:
  def __init__(self, dataset, model=None):
    self._dataset = dataset
    self._lock = threading.Lock()
    self._done = threading.Event()
    self._inferData = None
    self._inferResults = None
    self._model = model
    self._timesteps = 0 + self._model.timesteps
    self._prevSteps = []
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
      if not(data is None):
        if self._timesteps:
          arr = self._prevSteps + [data,]
          self._prevSteps = list(arr[-self._timesteps:]) # COPY of list
          self._inferData = self._prevSteps # same as self._prevSteps
        else:
          self._inferData = data
        pass
        
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
    info = {}
    if self._model.trainable and not(self._dataset is None):
      data = self._dataset.sample()
      if data:
        info = self._model.fit(data)
    return info
  
  def _infer(self, trainInfo):
    with self._lock:
      data = self._inferData
      self._inferData = None
    if data is None: return
    
    if not(len(data) == self._timesteps): return
    samples = [Utils.tracked2sample(x['tracked']) for x in data]
    samples = Utils.samples2inputs(samples)
    T = numpy.diff(samples['time'], 1)
    T = numpy.insert(T, 0, 0.0)
    samples['time'] = T.reshape((self._timesteps, 1))
    X = {k: x[None] for k, x in samples.items()} # (timesteps, ...) => (1, timesteps, ...)

    data = data[-1] # last step as current
    
    res = self._model(X)
    info = dict(trainInfo)
    if not(self._dataset is None):
      info['samples'] = len(self._dataset)
      
    with self._lock:
      self._inferResults = (res, data, info)
    return