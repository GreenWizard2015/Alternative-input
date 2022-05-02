import networks
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=5, threshold=77777777, linewidth=545556)
 
coords = np.linspace(0., 1.0, 100)
minD = (coords[1:] - coords[:-1]).min()
points = np.dstack(np.meshgrid(coords, coords)).reshape((-1, 2))

enricher = networks.PointsEnricher(points.shape[1:])
encoded = []
for i in range(0, len(points), 1000):
  encoded.append(enricher(points[i:i+1000])[0])
  continue

encoded = np.vstack(encoded)
print(encoded[:15])
print(encoded.shape)

uniqN = []
for i in range(encoded.shape[-1]):
  values = np.sort(np.unique(encoded[..., i]))
  diff = values[1:] - values[:-1]
  print(i, len(values), diff.min(), diff.max(), diff.mean())
  continue
# '''
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.plot(points[..., 0].ravel(), points[..., 1].ravel(), encoded[..., -1].ravel())
plt.show()
'''
from scipy.spatial.distance import cdist
# dist = cdist(encoded, encoded)
dist = np.sqrt(np.square(encoded[1:]- encoded[:-1]).sum(-1))
# dist = np.inf
print(dist.shape)
print((dist <= 0.).sum())
print(dist[dist > 0.].min())
print(dist[dist > 0.].mean())
print(dist.max())
x = plt.hist(dist.ravel(), 256, (.0, .75))
plt.vlines([minD], [0], [x[0].max()], colors='red')
plt.show()
'''