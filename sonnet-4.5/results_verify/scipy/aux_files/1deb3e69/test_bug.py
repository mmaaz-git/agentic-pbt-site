import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

print("=== Basic reproduction ===")
obs = np.array([[0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.]])

Z = linkage(obs, method='ward')
print(f"Linkage matrix Z:\n{Z}")

clusters = fcluster(Z, 2, criterion='maxclust')

print(f"Requested: 2 clusters")
print(f"Got: {len(np.unique(clusters))} cluster(s)")
print(f"Cluster assignments: {clusters}")

print("\n=== Extended reproduction showing the pattern ===")
obs = np.array([[0., 0.], [0., 0.], [0., 0.], [0., 0.]])
Z = linkage(obs, method='ward')
print(f"Linkage matrix Z:\n{Z}")

for k in [1, 2, 3, 4]:
    clusters = fcluster(Z, k, criterion='maxclust')
    n_actual = len(np.unique(clusters))
    print(f"k={k}: got {n_actual} clusters (expected {k})")
    print(f"  Cluster assignments: {clusters}")