import numpy as np
from scipy.cluster.vq import kmeans2

data = np.array([
    [1.0, 2.0],
    [1.0, 2.0],
    [1.0, 2.0],
    [1.0, 2.0],
])

try:
    centroid, label = kmeans2(data, 2, minit='random')
    print(f"Success: centroids shape = {centroid.shape}")
except np.linalg.LinAlgError as e:
    print(f"Crash: {e}")