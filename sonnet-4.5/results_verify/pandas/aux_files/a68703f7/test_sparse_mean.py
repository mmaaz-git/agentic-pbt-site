#!/usr/bin/env python3
"""Test script to reproduce the SparseArray.mean bug with NaN fill value"""

from pandas.arrays import SparseArray
import numpy as np
from hypothesis import given, strategies as st, assume
import math

# First, let's reproduce the basic bug
print("=" * 60)
print("BASIC BUG REPRODUCTION")
print("=" * 60)

arr = SparseArray([0.0, np.nan], fill_value=np.nan)
sparse_mean = arr.mean()
dense_mean = arr.to_dense().mean()

print(f"Array: {arr}")
print(f"Sparse mean: {sparse_mean}")
print(f"Dense mean: {dense_mean}")
print(f"Are they equal? {sparse_mean == dense_mean or (np.isnan(sparse_mean) and np.isnan(dense_mean))}")

# Let's also test NumPy's behavior with NaN
print("\n" + "=" * 60)
print("NUMPY BEHAVIOR WITH NaN")
print("=" * 60)

np_arr = np.array([0.0, np.nan])
np_mean = np_arr.mean()
print(f"NumPy array: {np_arr}")
print(f"NumPy mean: {np_mean}")

# Test with different configurations
print("\n" + "=" * 60)
print("ADDITIONAL TEST CASES")
print("=" * 60)

test_cases = [
    [1.0, np.nan],
    [np.nan, np.nan],
    [0.0, 1.0, np.nan],
    [np.nan, 1.0, 2.0, np.nan],
    [5.0],  # No NaN
]

for test_data in test_cases:
    sparse_arr = SparseArray(test_data, fill_value=np.nan)
    sparse_mean = sparse_arr.mean()
    dense_mean = sparse_arr.to_dense().mean()
    numpy_mean = np.array(test_data).mean()

    print(f"\nData: {test_data}")
    print(f"Sparse mean: {sparse_mean}")
    print(f"Dense mean: {dense_mean}")
    print(f"NumPy mean: {numpy_mean}")

    # Check if they match
    sparse_dense_match = sparse_mean == dense_mean or (np.isnan(sparse_mean) and np.isnan(dense_mean))
    dense_numpy_match = dense_mean == numpy_mean or (np.isnan(dense_mean) and np.isnan(numpy_mean))

    print(f"Sparse == Dense? {sparse_dense_match}")
    print(f"Dense == NumPy? {dense_numpy_match}")

# Now let's run the hypothesis test
print("\n" + "=" * 60)
print("HYPOTHESIS TEST")
print("=" * 60)

@st.composite
def sparse_arrays(draw):
    size = draw(st.integers(min_value=1, max_value=100))
    fill_value = np.nan
    elements = st.floats(allow_nan=True, allow_infinity=False)
    data = draw(st.lists(elements, min_size=size, max_size=size))
    return SparseArray(data, fill_value=fill_value)

@given(sparse_arrays())
def test_mean_matches_dense(arr):
    assume(len(arr) > 0)
    sparse_mean = arr.mean()
    dense_mean = arr.to_dense().mean()

    if isinstance(sparse_mean, float) and np.isnan(sparse_mean):
        assert np.isnan(dense_mean), f"Sparse is NaN but dense is {dense_mean}"
    else:
        assert math.isclose(sparse_mean, dense_mean, rel_tol=1e-9, abs_tol=1e-9), f"Sparse {sparse_mean} != Dense {dense_mean}"

# Run hypothesis test
try:
    test_mean_matches_dense()
    print("Hypothesis test PASSED - no failures found")
except AssertionError as e:
    print(f"Hypothesis test FAILED: {e}")
except Exception as e:
    print(f"Hypothesis test ERROR: {e}")

# Let's also check the internal structure of SparseArray
print("\n" + "=" * 60)
print("SPARSE ARRAY INTERNALS")
print("=" * 60)

arr = SparseArray([0.0, np.nan], fill_value=np.nan)
print(f"Array: {arr}")
print(f"sp_values: {arr.sp_values}")
print(f"sp_index: {arr.sp_index}")
print(f"fill_value: {arr.fill_value}")
print(f"_null_fill_value: {arr._null_fill_value}")

# Check what _valid_sp_values returns
print(f"_valid_sp_values: {arr._valid_sp_values}")