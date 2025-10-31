import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse.csgraph as csgraph

# Simple test to understand what's happening
graph = csr_matrix([[0, 1], [1, 0]], dtype=float)

# Get the LinearOperator
lap_lo = csgraph.laplacian(graph, form='lo')

# Test with 1D vector
test_vec_1d = np.array([1.0, 2.0])
print(f"Test vector 1D shape: {test_vec_1d.shape}, ndim: {test_vec_1d.ndim}")

# Let's trace through what the lambda function does
# From the source code, we have:
# _laplace(m, d) returns: lambda v: v * d[:, np.newaxis] - m @ v

# Get the graph sum (diagonal values)
graph_sum = np.array(graph.sum(axis=0)).ravel()
print(f"Graph sum: {graph_sum}")

# The diagonal is subtracted
graph_diagonal = graph.diagonal()
print(f"Graph diagonal: {graph_diagonal}")

diag = graph_sum - graph_diagonal
print(f"Diagonal for Laplacian: {diag}")

# Now simulate what the lambda does:
print("\nSimulating the lambda function behavior:")
v = test_vec_1d
print(f"v shape: {v.shape}")
print(f"diag[:, np.newaxis] shape: {diag[:, np.newaxis].shape}")
print(f"v * diag[:, np.newaxis] shape: {(v * diag[:, np.newaxis]).shape}")

result_part1 = v * diag[:, np.newaxis]
print(f"v * diag[:, np.newaxis] = \n{result_part1}")

result_part2 = graph @ v
print(f"graph @ v = {result_part2}")

final_result = result_part1 - result_part2
print(f"Final result shape: {final_result.shape}")
print(f"Final result:\n{final_result}")

# The issue is that when v is 1D, v * d[:, np.newaxis] creates a 2D array!
print("\n" + "="*60)
print("The problem:")
print("="*60)
print("When v is 1D (shape (n,)), the broadcasting v * d[:, np.newaxis]")
print("produces a 2D result of shape (n, n) instead of (n,)!")
print()
print("For 2D input (matmat):")
v2d = np.array([[1.0], [2.0]])
print(f"v2d shape: {v2d.shape}")
result_2d = v2d * diag[:, np.newaxis] - graph @ v2d
print(f"Result shape for 2D: {result_2d.shape}")
print(f"Result:\n{result_2d}")