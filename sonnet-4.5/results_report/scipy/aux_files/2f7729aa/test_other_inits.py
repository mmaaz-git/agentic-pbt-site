import numpy as np
from scipy.cluster.vq import kmeans2

# Create rank-deficient data where all features are perfectly correlated
data = np.array([[1.0, 1.0, 1.0],
                 [2.0, 2.0, 2.0],
                 [3.0, 3.0, 3.0],
                 [4.0, 4.0, 4.0],
                 [5.0, 5.0, 5.0]])

print("Testing different initialization methods with rank-deficient data\n")

for minit in ['points', '++', 'random']:
    print(f"Testing minit='{minit}':")
    try:
        codebook, labels = kmeans2(data, 2, minit=minit, iter=1)
        print(f"  SUCCESS - Got codebook shape {codebook.shape} and labels {labels}")
    except Exception as e:
        print(f"  FAILED - {type(e).__name__}: {e}")
    print()