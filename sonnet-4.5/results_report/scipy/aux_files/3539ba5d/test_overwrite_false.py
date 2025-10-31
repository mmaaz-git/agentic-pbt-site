import numpy as np
from scipy.sparse.csgraph import floyd_warshall

np.random.seed(0)
graph = np.random.rand(3, 3) * 10
graph_f = np.asfortranarray(graph, dtype=np.float64)

print("Testing F-contiguous array with overwrite=False:")
print("Original graph (F-contiguous):")
print(graph_f)
print(f"Diagonal: {np.diag(graph_f)}")

result = floyd_warshall(graph_f, directed=True, overwrite=False)

print("\nResult after floyd_warshall with overwrite=False:")
print(result)
print(f"Diagonal: {np.diag(result)}")
print(f"Is diagonal all zeros? {np.allclose(np.diag(result), 0)}")