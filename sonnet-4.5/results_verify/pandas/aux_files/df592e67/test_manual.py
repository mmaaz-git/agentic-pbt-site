from pandas.arrays import SparseArray
import numpy as np

sparse = SparseArray([1, 0, 0, 2], dtype=np.int64)
print(f"Original type: {type(sparse)}")

sparse_float = sparse.astype(np.float64)
print(f"After astype type: {type(sparse_float)}")
print(f"Is SparseArray: {isinstance(sparse_float, SparseArray)}")