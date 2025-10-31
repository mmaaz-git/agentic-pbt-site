import numpy as np
from scipy.cluster.vq import kmeans2

# Data with duplicate rows
data = np.array([
    [1.0, 2.0],
    [1.0, 2.0],
    [1.0, 2.0],
    [1.0, 2.0],
])

print("Testing different initialization methods on duplicate data:")
print(f"Data shape: {data.shape}")
print(f"Data:\n{data}\n")

# Test 'points' initialization
try:
    centroid, label = kmeans2(data, 2, minit='points')
    print(f"'points' method: SUCCESS, centroids shape = {centroid.shape}")
except Exception as e:
    print(f"'points' method: FAILED with {type(e).__name__}: {e}")

# Test '++' initialization
try:
    centroid, label = kmeans2(data, 2, minit='++')
    print(f"'++' method: SUCCESS, centroids shape = {centroid.shape}")
except Exception as e:
    print(f"'++' method: FAILED with {type(e).__name__}: {e}")

# Test 'random' initialization
try:
    centroid, label = kmeans2(data, 2, minit='random')
    print(f"'random' method: SUCCESS, centroids shape = {centroid.shape}")
except Exception as e:
    print(f"'random' method: FAILED with {type(e).__name__}: {e}")