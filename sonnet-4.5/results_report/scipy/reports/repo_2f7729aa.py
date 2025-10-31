import numpy as np
from scipy.cluster.vq import kmeans2

# Create rank-deficient data where all features are perfectly correlated
data = np.array([[1.0, 1.0, 1.0],
                 [2.0, 2.0, 2.0],
                 [3.0, 3.0, 3.0],
                 [4.0, 4.0, 4.0],
                 [5.0, 5.0, 5.0]])

print("Testing scipy.cluster.vq.kmeans2 with rank-deficient data")
print("Data shape:", data.shape)
print("Data:\n", data)
print("\nAttempting kmeans2 with k=2, minit='random'...")

try:
    codebook, labels = kmeans2(data, 2, minit='random')
    print("Success! Codebook:", codebook)
    print("Labels:", labels)
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")