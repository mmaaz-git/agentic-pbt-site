import numpy as np
from scipy.cluster.vq import kmeans2

data = np.array([[1.0, 1.0, 1.0],
                 [2.0, 2.0, 2.0],
                 [3.0, 3.0, 3.0],
                 [4.0, 4.0, 4.0],
                 [5.0, 5.0, 5.0]])

print("Testing kmeans2 with rank-deficient data (all features perfectly correlated)")
print(f"Input data shape: {data.shape}")
print(f"Input data:\n{data}")

try:
    codebook, labels = kmeans2(data, 2, minit='random')
    print("Success! kmeans2 did not crash.")
    print(f"Codebook:\n{codebook}")
    print(f"Labels: {labels}")
except np.linalg.LinAlgError as e:
    print(f"Error: {e}")
    print("kmeans2 crashed with LinAlgError as reported in the bug.")