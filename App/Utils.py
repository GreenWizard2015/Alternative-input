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

def normalized(a, axis=-1, order=2):
  l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
  l2[l2==0] = 1
  return a / np.expand_dims(l2, axis)

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
  
