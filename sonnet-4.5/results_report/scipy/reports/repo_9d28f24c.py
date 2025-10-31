import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse.csgraph as csgraph

# Create a simple 2x2 symmetric graph
graph = csr_matrix([[0, 1], [1, 0]], dtype=float)

# Create LinearOperator with form='lo'
lap_lo = csgraph.laplacian(graph, form='lo')

# Create test vector
test_vec = np.array([1.0, 2.0])

# Attempt matrix-vector multiplication - this should crash
try:
    result = lap_lo @ test_vec
    print(f"Result: {result}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")

# Also test with matvec directly
print("\nTesting with matvec directly:")
try:
    result = lap_lo.matvec(test_vec)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")

# Compare with form='array' which should work
print("\nComparing with form='array':")
lap_array = csgraph.laplacian(graph, form='array')
result_array = lap_array @ test_vec
print(f"Result from array form: {result_array}")
print(f"Shape: {result_array.shape}")

# Test with 2D input (which should work)
print("\nTesting with 2D input:")
test_vec_2d = np.array([[1.0], [2.0]])
try:
    result_2d = lap_lo @ test_vec_2d
    print(f"Result from 2D input: {result_2d}")
    print(f"Shape: {result_2d.shape}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")