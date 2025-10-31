import numpy as np
from scipy.sparse.csgraph import floyd_warshall

np.random.seed(0)
graph = np.random.rand(3, 3) * 10
graph_f = np.asfortranarray(graph, dtype=np.float64)

print("Original graph (F-contiguous):")
print(graph_f)
print(f"Flags: C_CONTIGUOUS={graph_f.flags['C_CONTIGUOUS']}, F_CONTIGUOUS={graph_f.flags['F_CONTIGUOUS']}")
print(f"Diagonal: {np.diag(graph_f)}")

result = floyd_warshall(graph_f, directed=True, overwrite=True)

print("\nResult after floyd_warshall with overwrite=True:")
print(result)
print(f"Diagonal: {np.diag(result)}")
print(f"\nExpected diagonal: [0, 0, 0]")
print(f"Is diagonal all zeros? {np.allclose(np.diag(result), 0)}")
print(f"Is result identical to input? {np.array_equal(result, graph_f)}")

# Also test with C-contiguous array for comparison
graph_c = np.ascontiguousarray(graph, dtype=np.float64)
print("\n--- For comparison: C-contiguous array ---")
print("Original graph (C-contiguous):")
print(graph_c)
print(f"Flags: C_CONTIGUOUS={graph_c.flags['C_CONTIGUOUS']}, F_CONTIGUOUS={graph_c.flags['F_CONTIGUOUS']}")
print(f"Diagonal: {np.diag(graph_c)}")

result_c = floyd_warshall(graph_c, directed=True, overwrite=True)
print("\nResult after floyd_warshall with overwrite=True:")
print(result_c)
print(f"Diagonal: {np.diag(result_c)}")
print(f"Is diagonal all zeros? {np.allclose(np.diag(result_c), 0)}")