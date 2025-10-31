import numpy as np
from scipy.cluster.vq import kmeans2

data = np.array([[1.0, 1.0, 1.0],
                 [2.0, 2.0, 2.0],
                 [3.0, 3.0, 3.0],
                 [4.0, 4.0, 4.0],
                 [5.0, 5.0, 5.0]])

print("Testing kmeans2 with rank-deficient data using different initialization methods\n")

# Test different initialization methods
init_methods = ['points', '++', 'random']

for minit in init_methods:
    print(f"Testing minit='{minit}':")
    try:
        codebook, labels = kmeans2(data, 2, minit=minit, iter=1, seed=42)
        print(f"  SUCCESS - No crash")
        print(f"  Labels: {labels}")
    except Exception as e:
        print(f"  FAILED - {type(e).__name__}: {e}")
    print()

# Test with matrix initialization
print("Testing with matrix initialization:")
init_matrix = np.array([[1.0, 1.0, 1.0], [5.0, 5.0, 5.0]])
try:
    codebook, labels = kmeans2(data, init_matrix, iter=1)
    print(f"  SUCCESS - No crash")
    print(f"  Labels: {labels}")
except Exception as e:
    print(f"  FAILED - {type(e).__name__}: {e}")