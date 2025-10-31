import numpy as np
from scipy.sparse.csgraph import csgraph_from_dense

# Find the exact threshold where values start being kept
test_values = []
for exp in range(-15, -5):
    for mantissa in [1, 2, 5]:
        test_values.append(mantissa * 10**exp)

print("Finding the threshold where values are kept...")
print("-" * 50)

results = []
for val in sorted(test_values):
    graph = np.array([[0.0, val], [0.0, 0.0]])
    sparse = csgraph_from_dense(graph, null_value=0)
    kept = sparse.nnz > 0
    results.append((val, kept))
    print(f"Value: {val:.2e} - {'KEPT' if kept else 'DROPPED'}")

print("\n" + "=" * 50)
print("Threshold appears to be between:")
for i in range(len(results) - 1):
    if not results[i][1] and results[i+1][1]:
        print(f"  {results[i][0]:.2e} (dropped) and {results[i+1][0]:.2e} (kept)")
        break