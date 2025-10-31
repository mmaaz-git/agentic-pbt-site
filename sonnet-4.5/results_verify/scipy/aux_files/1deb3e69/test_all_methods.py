import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

# Test with identical observations
obs = np.array([[0., 0.], [0., 0.], [0., 0.], [0., 0.]])

methods = ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']

print("Testing all linkage methods with identical observations:")
print("=" * 60)

for method in methods:
    print(f"\nMethod: {method}")
    try:
        Z = linkage(obs, method=method)
        print(f"  Linkage matrix: {Z}")
        print("  Clusters requested vs actual:")
        for k in [1, 2, 3, 4]:
            clusters = fcluster(Z, k, criterion='maxclust')
            n_actual = len(np.unique(clusters))
            status = "✓" if n_actual == k else "✗"
            print(f"    k={k}: got {n_actual} clusters (expected {k}) {status}")
    except Exception as e:
        print(f"  Error: {e}")

# Now test with tiny perturbations
print("\n" + "=" * 60)
print("Testing with tiny perturbations (1e-15):")
print("=" * 60)

np.random.seed(42)
obs_perturbed = np.array([[0., 0.], [0., 0.], [0., 0.], [0., 0.]]) + np.random.randn(4, 2) * 1e-15

for method in ['ward', 'single', 'complete']:
    print(f"\nMethod: {method}")
    try:
        Z = linkage(obs_perturbed, method=method)
        print(f"  Linkage matrix distances: {Z[:, 2]}")
        print("  Clusters requested vs actual:")
        for k in [1, 2, 3, 4]:
            clusters = fcluster(Z, k, criterion='maxclust')
            n_actual = len(np.unique(clusters))
            status = "✓" if n_actual == k else "✗"
            print(f"    k={k}: got {n_actual} clusters (expected {k}) {status}")
    except Exception as e:
        print(f"  Error: {e}")