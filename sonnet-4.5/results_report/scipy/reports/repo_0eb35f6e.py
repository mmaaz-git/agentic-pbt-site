import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

# Test case with identical observations
obs = np.array([[0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.]])

Z = linkage(obs, method='ward')
print("Linkage matrix Z:")
print(Z)
print()

# Request 2 clusters but gets only 1
clusters = fcluster(Z, 2, criterion='maxclust')

print(f"Requested: 2 clusters")
print(f"Got: {len(np.unique(clusters))} cluster(s)")
print(f"Cluster assignments: {clusters}")
print()

# Extended test showing the pattern with 4 identical observations
print("Extended test with 4 identical observations:")
obs4 = np.array([[0., 0.], [0., 0.], [0., 0.], [0., 0.]])
Z4 = linkage(obs4, method='ward')

for k in [1, 2, 3, 4]:
    clusters = fcluster(Z4, k, criterion='maxclust')
    n_actual = len(np.unique(clusters))
    status = "✓" if n_actual == k else "✗"
    print(f"k={k}: got {n_actual} clusters (expected {k}) {status}")