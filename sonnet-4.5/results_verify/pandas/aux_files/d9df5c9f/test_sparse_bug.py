#!/usr/bin/env python3
"""Test the reported SparseArray.astype bug."""

from pandas.core.sparse.api import SparseArray
import numpy as np
from hypothesis import given
import hypothesis.extra.numpy as npst

# First, run the simple reproduction case
print("=" * 60)
print("Testing simple reproduction case:")
print("=" * 60)

sparse = SparseArray([1.0, 0.0, 2.0])
print(f"Created SparseArray: {sparse}")
print(f"Type of sparse: {type(sparse)}")

result = sparse.astype(np.int64)
print(f"\nAfter calling astype(np.int64):")
print(f"Result: {result}")
print(f"Type of result: {type(result)}")
print(f"Expected type: <class 'pandas.core.arrays.sparse.array.SparseArray'>")
print(f"Is SparseArray? {isinstance(result, SparseArray)}")

# Now test the hypothesis property-based test
print("\n" + "=" * 60)
print("Testing hypothesis property-based test:")
print("=" * 60)

@given(npst.arrays(dtype=np.float64, shape=npst.array_shapes(min_dims=1, max_dims=1)))
def test_astype_preserves_type(arr):
    sparse = SparseArray(arr)
    result = sparse.astype(np.int64)
    assert isinstance(result, SparseArray), f"astype should return SparseArray, got {type(result)}"

# Run with the specific failing example
test_arr = np.array([0.])
print(f"Testing with specific failing input: {test_arr}")
sparse_test = SparseArray(test_arr)
print(f"Created SparseArray: {sparse_test}")
result_test = sparse_test.astype(np.int64)
print(f"Result after astype(np.int64): {result_test}")
print(f"Type: {type(result_test)}")
print(f"Is SparseArray? {isinstance(result_test, SparseArray)}")

# Test with SparseDtype
print("\n" + "=" * 60)
print("Testing with SparseDtype:")
print("=" * 60)
from pandas import SparseDtype
sparse2 = SparseArray([1.0, 0.0, 2.0])
result2 = sparse2.astype(SparseDtype(np.int64))
print(f"Result with SparseDtype(np.int64): {result2}")
print(f"Type: {type(result2)}")
print(f"Is SparseArray? {isinstance(result2, SparseArray)}")

# Check what the docstring actually says
print("\n" + "=" * 60)
print("Checking docstring:")
print("=" * 60)
print(SparseArray.astype.__doc__)