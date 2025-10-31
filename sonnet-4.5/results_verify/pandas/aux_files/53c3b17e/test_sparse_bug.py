#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

import numpy as np
from pandas.arrays import SparseArray

# Test 1: The reported bug case
print("Test 1: SparseArray([0], fill_value=0)")
try:
    arr = SparseArray([0], fill_value=0)
    print(f"  Array created: {arr}")
    print(f"  Array.sp_values: {arr.sp_values}")
    print(f"  Array.fill_value: {arr.fill_value}")
    result = arr.argmin()
    print(f"  argmin() result: {result}")
except Exception as e:
    print(f"  ERROR on argmin(): {type(e).__name__}: {e}")

try:
    arr = SparseArray([0], fill_value=0)
    result = arr.argmax()
    print(f"  argmax() result: {result}")
except Exception as e:
    print(f"  ERROR on argmax(): {type(e).__name__}: {e}")

# Test 2: Compare with NumPy behavior
print("\nTest 2: Compare with NumPy")
np_arr = np.array([0])
print(f"  NumPy array: {np_arr}")
print(f"  NumPy argmin(): {np_arr.argmin()}")
print(f"  NumPy argmax(): {np_arr.argmax()}")

# Test 3: Dense array comparison
print("\nTest 3: Compare SparseArray with its dense version")
arr = SparseArray([0], fill_value=0)
dense = arr.to_dense()
print(f"  Dense array: {dense}")
print(f"  Dense argmin(): {dense.argmin()}")
print(f"  Dense argmax(): {dense.argmax()}")

# Test 4: Multiple elements all equal to fill_value
print("\nTest 4: SparseArray([0, 0, 0], fill_value=0)")
try:
    arr = SparseArray([0, 0, 0], fill_value=0)
    print(f"  Array created: {arr}")
    print(f"  Array.sp_values: {arr.sp_values}")
    result = arr.argmin()
    print(f"  argmin() result: {result}")
except Exception as e:
    print(f"  ERROR on argmin(): {type(e).__name__}: {e}")

# Test 5: Mixed values (some equal to fill_value)
print("\nTest 5: SparseArray([0, 1, 0], fill_value=0)")
try:
    arr = SparseArray([0, 1, 0], fill_value=0)
    print(f"  Array created: {arr}")
    print(f"  Array.sp_values: {arr.sp_values}")
    result_min = arr.argmin()
    result_max = arr.argmax()
    print(f"  argmin() result: {result_min}")
    print(f"  argmax() result: {result_max}")
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")

# Test 6: Run the hypothesis test
print("\nTest 6: Running the hypothesis test from the bug report")
from hypothesis import given, strategies as st

@given(st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=50))
def test_argmin_argmax_match_dense(data):
    arr = SparseArray(data, fill_value=0)
    dense = arr.to_dense()

    try:
        sparse_min = arr.argmin()
        dense_min = dense.argmin()
        assert sparse_min == dense_min, f"argmin mismatch: sparse={sparse_min}, dense={dense_min}"

        sparse_max = arr.argmax()
        dense_max = dense.argmax()
        assert sparse_max == dense_max, f"argmax mismatch: sparse={sparse_max}, dense={dense_max}"
    except Exception as e:
        print(f"  Failed on data={data}")
        print(f"  Error: {type(e).__name__}: {e}")
        raise

# Run with the specific failing case
print("  Testing with data=[0]:")
try:
    test_argmin_argmax_match_dense([0])
    print("  PASSED")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")