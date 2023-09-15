import cv2, pygame
import numpy as np

class Colors:
  BLACK = (0, 0, 0)
  SILVER = (192, 192, 192)
  WHITE = (255, 255, 255)
  BLUE = (0, 0, 255)
  GREEN = (0, 255, 0)
  RED = (255, 0, 0)
  PURPLE = (255, 0, 255)
# take colors from matplotlib and add to Colors
def _makeColors():
  import matplotlib.colors as mcolors
  for name, hex in mcolors.cnames.items():
    setattr(Colors, name.upper(), tuple(int(hex[i:i+2], 16) for i in (1, 3, 5)))
  return
_makeColors()
# add all colors to a list, if its RGB. use __dict__.values()
Colors.asList = [rgb for rgb in Colors.__dict__.values() if type(rgb) == tuple and len(rgb) == 3]
################################################
def normalized(a, axis=-1, order=2):
  l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
  l2[l2==0] = 1
  return a / np.expand_dims(l2, axis)

def densityToSurface(cv2Image):
  size = cv2Image.shape[:-1]
  fmt = 'RGB'
  surface = pygame.image.frombuffer(cv2Image.flatten(), size, fmt)
  return surface.convert()

def cv2ImageToSurface(cv2Image):
  if cv2Image.dtype.name == 'uint16':
    cv2Image = (cv2Image / 256).astype('uint8')

  size = cv2Image.shape[1::-1]
  fmt = None
  if len(cv2Image.shape) == 2:
    cv2Image = np.repeat(cv2Image.reshape(size[1], size[0], 1), 3, axis = 2)
    fmt = 'RGB'
  else:
    fmt = 'RGBA' if cv2Image.shape[2] == 4 else 'RGB'
    cv2Image[:, :, [0, 2]] = cv2Image[:, :, [2, 0]]
  
  surface = pygame.image.frombuffer(cv2Image.flatten(), size, fmt)
  return surface.convert_alpha() if fmt == 'RGBA' else surface.convert()

def rotate(vector, rads):
  return np.array([
    np.cos(rads) * vector[0] - np.sin(rads) * vector[1],
    np.sin(rads) * vector[0] + np.cos(rads) * vector[1],
  ])
  
def numpyToSurfaceBind(array, surface):
  surf = pygame.surfarray.pixels3d(surface)
  WH = surf.shape[:2]
  array = cv2.resize(array, tuple(WH))
  if 2 == len(array.shape): array = array.reshape((*array.shape, 1)) # H x W -> H x W x 1
  if 1 == array.shape[-1]: array = np.repeat(array, 3, axis=-1) # grayscale -> RGB
  array = np.swapaxes(array, 0, 1) # H x W x C -> W x H x C
  surf[:, :, :] = array
  del surf # release surface
  return