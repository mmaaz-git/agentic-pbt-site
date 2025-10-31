import numpy as np
from scipy.sparse.csgraph._validation import validate_graph

G = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=np.int32)

result_float32 = validate_graph(G, directed=True, dtype=np.float32)
result_float64 = validate_graph(G, directed=True, dtype=np.float64)
result_int32 = validate_graph(G, directed=True, dtype=np.int32)

print(f"Result with dtype=np.float32: {result_float32.dtype}")
print(f"Result with dtype=np.float64: {result_float64.dtype}")
print(f"Result with dtype=np.int32: {result_int32.dtype}")

# Check if the dtype parameter is being honored
print(f"\nExpected dtype=np.float32, got {result_float32.dtype}: Match = {result_float32.dtype == np.float32}")
print(f"Expected dtype=np.float64, got {result_float64.dtype}: Match = {result_float64.dtype == np.float64}")
print(f"Expected dtype=np.int32, got {result_int32.dtype}: Match = {result_int32.dtype == np.int32}")

assert result_float32.dtype == np.float32, f"Expected float32 but got {result_float32.dtype}"