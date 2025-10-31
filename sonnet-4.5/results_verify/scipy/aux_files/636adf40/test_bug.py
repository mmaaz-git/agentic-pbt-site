import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse.csgraph as csgraph

# Test the simple reproduction case from the bug report
print("Testing simple reproduction case:")
graph = csr_matrix([[0, 1], [1, 0]], dtype=float)
lap_lo = csgraph.laplacian(graph, form='lo')
test_vec = np.array([1.0, 2.0])

print("Graph matrix:")
print(graph.toarray())
print("\nLinearOperator:")
print(lap_lo)
print(f"Shape: {lap_lo.shape}")

print("\nTest vector:")
print(test_vec)
print(f"Test vector shape: {test_vec.shape}")

try:
    result = lap_lo @ test_vec
    print("\nResult of lap_lo @ test_vec:")
    print(result)
    print(f"Result shape: {result.shape}")
except Exception as e:
    print(f"\nError occurred: {type(e).__name__}: {e}")

# Let's also test with form='array' to see what the expected behavior is
print("\n\nTesting with form='array' for comparison:")
lap_array = csgraph.laplacian(graph, form='array')
print("Laplacian matrix (array form):")
print(lap_array.toarray())

result_array = lap_array @ test_vec
print("\nResult of lap_array @ test_vec:")
print(result_array)
print(f"Result shape: {result_array.shape}")

# Test the matvec method directly
print("\n\nTesting matvec method directly:")
try:
    result_matvec = lap_lo.matvec(test_vec)
    print("Result of lap_lo.matvec(test_vec):")
    print(result_matvec)
    print(f"Result shape: {result_matvec.shape}")
except Exception as e:
    print(f"Error occurred with matvec: {type(e).__name__}: {e}")

# Test with 2D input as well
print("\n\nTesting with 2D input (matmat):")
test_mat = np.array([[1.0], [2.0]])
print(f"Test matrix shape: {test_mat.shape}")
try:
    result_matmat = lap_lo @ test_mat
    print("Result of lap_lo @ test_mat:")
    print(result_matmat)
    print(f"Result shape: {result_matmat.shape}")
except Exception as e:
    print(f"Error occurred with 2D input: {type(e).__name__}: {e}")