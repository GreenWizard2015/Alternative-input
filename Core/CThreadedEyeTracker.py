from Core.CEyeTracker import CEyeTracker
import threading
import time

class CThreadedEyeTracker:
  def __init__(self):
    self._lock = threading.Lock()
    self._done = threading.Event()
    self._results = None
    return

  def __enter__(self):
    self._tracker = CEyeTracker()
    self._tracker.__enter__()
    
    self._thread = threading.Thread(target=self._trackLoop)
    self._thread.start()
    return self

  def __exit__(self, type, value, traceback):
    self._done.set()
    self._thread.join()
    self._tracker.__exit__(type, value, traceback)
    return

  def track(self):
    with self._lock:
      res = self._results
      self._results = None
    return res
  
  def _trackLoop(self):
    while not self._done.isSet():
      res = self._tracker.track()
      with self._lock:
        self._results = res
      time.sleep(0) # idle
      continue
    return