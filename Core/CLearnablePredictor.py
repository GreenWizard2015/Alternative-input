import threading
import Core.Utils as Utils
import numpy

class CLearnablePredictor:
  def __init__(self, model, fps=30):
    self._lock = threading.Lock()
    self._done = threading.Event()
    self._inferData = None
    self._inferResults = None
    self._model = model
    self._timesteps = self._model.timesteps
    self._prevSteps = []
    self._fps = fps
    return

  def __enter__(self):
    self._thread = threading.Thread(target=self._loop)
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
  
  def _loop(self):
    while not self._done.wait(1.0 / self._fps):
      self._infer()
      continue
    return
  
  def _infer(self):
    with self._lock:
      data = self._inferData
      self._inferData = None
    if data is None: return
    
    if not(len(data) == self._timesteps): return
    samples = [Utils.tracked2sample(x) for x in data]
    samples = Utils.samples2inputs(samples)
    T = numpy.diff(samples['time'], 1)
    T = numpy.insert(T, 0, 0.0)
    samples['time'] = T.reshape((self._timesteps, 1))
    X = {k: x[None] for k, x in samples.items()} # (timesteps, ...) => (1, timesteps, ...)

    data = data[-1] # last step as current
    res = self._model(X)

    with self._lock:
      self._inferResults = (res, data, {})
    return
  
  @property
  def canPredict(self):
    return True