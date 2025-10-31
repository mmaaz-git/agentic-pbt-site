import numpy as np
from scipy.cluster.vq import kmeans2

# Test case from bug report
np.random.seed(0)
data = np.random.randn(5, 5)

print(f"Data shape: {data.shape}")
print(f"Testing kmeans2 with k=2, minit='random'...")

try:
    centroids, labels = kmeans2(data, 2, minit='random')
    print("Success! No error occurred.")
    print(f"Centroids shape: {centroids.shape}")
    print(f"Labels length: {len(labels)}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")