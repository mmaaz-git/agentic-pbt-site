#!/usr/bin/env python3
"""Test the SparseArray astype() bug"""

from pandas.arrays import SparseArray
from hypothesis import given, strategies as st
import numpy as np

# First, let's run the simple reproduction case
print("=== Simple Reproduction ===")
arr = SparseArray([1, 2, 3], dtype=np.int64)
result = arr.astype(np.float64)

print(f'Type of result: {type(result)}')
print(f'Result value: {result}')
print(f'Is SparseArray: {isinstance(result, SparseArray)}')
print(f'Is ndarray: {isinstance(result, np.ndarray)}')

# Let's also check what happens with SparseDtype
print("\n=== With SparseDtype ===")
from pandas import SparseDtype
sparse_dtype = SparseDtype(np.float64)
result2 = arr.astype(sparse_dtype)
print(f'Type of result2: {type(result2)}')
print(f'Is SparseArray: {isinstance(result2, SparseArray)}')

# Now let's run the property-based test
print("\n=== Running Property-Based Test ===")

@given(st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=50))
def test_sparse_array_astype_roundtrip(values):
    arr = SparseArray(values, dtype=np.int64)
    arr_float = arr.astype(np.float64)
    arr_int = arr_float.astype(np.int64)

    # The test expects to_dense() method on arr_float and arr_int
    # but if arr_float is an ndarray, it won't have to_dense()
    try:
        assert np.array_equal(arr.to_dense(), arr_int.to_dense()), \
            f"astype roundtrip failed: {arr.to_dense()} != {arr_int.to_dense()}"
    except AttributeError as e:
        print(f"AttributeError on values {values[:5]}...: {e}")
        return False
    return True

# Run the test with a simple example
try:
    test_sparse_array_astype_roundtrip([1, 2, 3])
    print("Test passed with [1, 2, 3]")
except Exception as e:
    print(f"Test failed with [1, 2, 3]: {e}")