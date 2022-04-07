import threading

class CDataset:
  def __init__(self):
    self._lock = threading.Lock()
    return

  def __enter__(self):
    return self

  def __exit__(self, type, value, traceback):
    return

  def store(self, eye, goal):
    with self._lock:
      pass
    return

  def pull(self):
    with self._lock:
      pass
    return