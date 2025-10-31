import numpy as np
from scipy.cluster.vq import kmeans2

# Test with various seeds to see when it fails
seeds_to_test = [0, 1, 2, 3, 4, 5, 10, 42, 100]

for seed in seeds_to_test:
    np.random.seed(seed)
    data = np.random.randn(5, 5)

    print(f"Testing with seed={seed}, data shape: {data.shape}")

    try:
        centroids, labels = kmeans2(data, 2, minit='random')
        print(f"  Success! Centroids shape: {centroids.shape}")
    except Exception as e:
        print(f"  Failed with {type(e).__name__}: {e}")