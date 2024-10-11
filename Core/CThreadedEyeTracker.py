from Core.CEyeTracker import CEyeTracker
import threading

class CThreadedEyeTracker:
  def __init__(self, fps=30, webcam=0):
    self._lock = threading.Lock()
    self._done = threading.Event()
    self._results = None
    self._fps = fps
    self._webcam = webcam
    return

  def __enter__(self):
    self._tracker = CEyeTracker(webcam=self._webcam)
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
    # wait for the event to be set up to 1/fps seconds
    while not self._done.wait(1.0 / self._fps):
      res = self._tracker.track()
      with self._lock:
        self._results = res
      continue
    return