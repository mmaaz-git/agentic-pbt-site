#!/usr/bin/env python3
"""Test the SparseArray cumsum bug report"""

import sys
import traceback
import pandas.arrays
import numpy as np

# Test 1: Basic reproduction from bug report
print("Test 1: Basic reproduction")
try:
    sparse = pandas.arrays.SparseArray([1, 0, 2, 0, 3], fill_value=0)
    result = sparse.cumsum()
    print(f"Success: {result}")
except RecursionError as e:
    print(f"RecursionError occurred as reported: {e}")
    traceback.print_exc(limit=10)  # Limit traceback to see the pattern
except Exception as e:
    print(f"Unexpected error: {e}")
    traceback.print_exc()

print("\n" + "="*50 + "\n")

# Test 2: Check what fill_value=0 does
print("Test 2: Understanding fill_value=0")
sparse = pandas.arrays.SparseArray([1, 0, 2, 0, 3], fill_value=0)
print(f"SparseArray: {sparse}")
print(f"Fill value: {sparse.fill_value}")
print(f"_null_fill_value: {sparse._null_fill_value}")
print(f"Dense representation: {sparse.to_dense()}")

print("\n" + "="*50 + "\n")

# Test 3: Test with null fill value (NaN)
print("Test 3: Test with NaN as fill_value")
try:
    sparse_nan = pandas.arrays.SparseArray([1, np.nan, 2, np.nan, 3], fill_value=np.nan)
    print(f"SparseArray: {sparse_nan}")
    print(f"Fill value: {sparse_nan.fill_value}")
    print(f"_null_fill_value: {sparse_nan._null_fill_value}")
    result_nan = sparse_nan.cumsum()
    print(f"Cumsum result: {result_nan}")
    print(f"Dense cumsum: {sparse_nan.to_dense().cumsum()}")
except Exception as e:
    print(f"Error with NaN fill_value: {e}")
    traceback.print_exc()

print("\n" + "="*50 + "\n")

# Test 4: Hypothesis test from bug report
print("Test 4: Property-based test")
from hypothesis import given, strategies as st, settings

test_counter = 0
test_failures = 0

@given(st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=30))
@settings(max_examples=10)  # Reduced for quicker testing
def test_sparse_array_cumsum_length(data):
    global test_counter, test_failures
    test_counter += 1
    try:
        sparse = pandas.arrays.SparseArray(data, fill_value=0)
        cumsum_result = sparse.cumsum()
        assert len(cumsum_result) == len(sparse)
        print(f"  Test {test_counter}: PASS with data length {len(data)}")
    except RecursionError:
        test_failures += 1
        print(f"  Test {test_counter}: FAIL (RecursionError) with data={data[:5]}...")
    except Exception as e:
        test_failures += 1
        print(f"  Test {test_counter}: FAIL ({type(e).__name__}) with data={data[:5]}...")

try:
    test_sparse_array_cumsum_length()
    print(f"\nTests completed: {test_counter} total, {test_failures} failures")
except Exception as e:
    print(f"\nTest suite failed: {e}")

print("\n" + "="*50 + "\n")

# Test 5: Check what the code at line 1550 actually does
print("Test 5: Understanding line 1550 behavior")
sparse = pandas.arrays.SparseArray([1, 0, 2, 0, 3], fill_value=0)
print(f"Original sparse array: {sparse}")
print(f"Type: {type(sparse)}")

dense = sparse.to_dense()
print(f"\nAfter to_dense(): {dense}")
print(f"Type: {type(dense)}")

wrapped = pandas.arrays.SparseArray(dense)
print(f"\nAfter wrapping in SparseArray: {wrapped}")
print(f"Type: {type(wrapped)}")

# What should happen vs what does happen
print("\n\nWhat SHOULD happen: dense.cumsum() then wrap")
correct_result = pandas.arrays.SparseArray(dense.cumsum())
print(f"Correct result: {correct_result}")

print("\n\nWhat DOES happen in line 1550: wrap then cumsum")
print("SparseArray(self.to_dense()).cumsum() causes infinite recursion")