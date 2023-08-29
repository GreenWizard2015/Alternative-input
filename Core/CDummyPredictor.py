class CDummyPredictor:
  def __enter__(self):
    return self

  def __exit__(self, type, value, traceback):
    return

  def async_infer(self, data):
    return None
  
  @property
  def canPredict(self):
    return False