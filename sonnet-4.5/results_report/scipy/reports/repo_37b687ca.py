import numpy as np
from scipy.cluster.vq import kmeans2

np.random.seed(0)
data = np.random.randn(5, 5)

centroids, labels = kmeans2(data, 2, minit='random')