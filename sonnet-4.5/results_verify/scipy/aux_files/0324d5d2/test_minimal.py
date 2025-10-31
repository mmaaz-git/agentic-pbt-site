import numpy as np
from scipy.cluster.vq import kmeans2

obs = np.array([[0., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])

print("Attempting to run kmeans2 with default 'random' initialization...")
try:
    centroids, labels = kmeans2(obs, 2)
    print(f"Success! Centroids shape: {centroids.shape}, Labels: {labels}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")

# Let's try with different initialization methods
print("\nTrying with 'points' initialization...")
try:
    centroids, labels = kmeans2(obs, 2, minit='points')
    print(f"Success! Centroids shape: {centroids.shape}, Labels: {labels}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")

print("\nTrying with '++' initialization...")
try:
    centroids, labels = kmeans2(obs, 2, minit='++')
    print(f"Success! Centroids shape: {centroids.shape}, Labels: {labels}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")