from pandas.arrays import SparseArray
import numpy as np

sparse = SparseArray([1, 0, 0, 2], dtype=np.int64)
print(f"Original type: {type(sparse)}")
print(f"Original array: {sparse}")

sparse_float = sparse.astype(np.float64)
print(f"\nAfter astype(np.float64):")
print(f"Result type: {type(sparse_float)}")
print(f"Is SparseArray: {isinstance(sparse_float, SparseArray)}")
print(f"Result values: {sparse_float}")

# According to the documentation, astype() should always return a SparseArray
# But we're getting a numpy.ndarray instead