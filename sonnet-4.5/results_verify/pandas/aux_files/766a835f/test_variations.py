import numpy as np
from pandas.arrays import SparseArray
from pandas import SparseDtype

sparse = SparseArray([0, 0, 1, 2], fill_value=0)
print("Original SparseArray:", type(sparse))
print("Original dtype:", sparse.dtype)
print()

# Test different dtype specifications
test_cases = [
    ("np.float64", np.float64),
    ("'float64'", 'float64'),
    ("np.int32", np.int32),
    ("'int32'", 'int32'),
    ("SparseDtype('float64')", SparseDtype('float64')),
    ("SparseDtype(np.float64)", SparseDtype(np.float64)),
    ("'Sparse[float64]'", 'Sparse[float64]'),
    ("'Sparse[float64, 0]'", 'Sparse[float64, 0]'),
]

for name, dtype in test_cases:
    try:
        result = sparse.astype(dtype)
        result_type = type(result).__name__
        is_sparse = isinstance(result, SparseArray)
        print(f"{name:30} -> {result_type:20} (SparseArray: {is_sparse})")
    except Exception as e:
        print(f"{name:30} -> ERROR: {e}")