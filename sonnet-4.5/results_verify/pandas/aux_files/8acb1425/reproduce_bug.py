#!/usr/bin/env python3
"""Reproduce the SparseArray.astype bug"""

from pandas.arrays import SparseArray
import numpy as np
from hypothesis import given, strategies as st

print("=" * 60)
print("Reproducing SparseArray.astype bug")
print("=" * 60)

# Test 1: Basic reproduction
print("\n1. Basic reproduction test:")
sparse_int = SparseArray([0, 0, 1, 2])
print(f"Original SparseArray: {sparse_int}")
print(f"Original type: {type(sparse_int)}")

result = sparse_int.astype(np.float64)
print(f"\nAfter astype(np.float64):")
print(f"Result: {result}")
print(f"Type of result: {type(result)}")
print(f"Is SparseArray: {isinstance(result, SparseArray)}")

try:
    assert isinstance(result, SparseArray)
    print("✓ Assertion passed: result is a SparseArray")
except AssertionError:
    print("✗ Assertion failed: result is NOT a SparseArray (expected based on docstring)")

# Test 2: Property-based test
print("\n" + "=" * 60)
print("2. Property-based test with Hypothesis:")

test_passed = True
failing_examples = []

@given(st.lists(st.integers(min_value=0, max_value=10), min_size=1, max_size=10))
def test_astype_returns_sparse_array(data):
    global test_passed, failing_examples
    sparse_int = SparseArray(data)
    result = sparse_int.astype(np.float64)

    if not isinstance(result, SparseArray):
        test_passed = False
        failing_examples.append(data)
        print(f"✗ Failed for data={data}, got type {type(result)}")

# Run hypothesis test
try:
    test_astype_returns_sparse_array()
except:
    pass

if test_passed:
    print("✓ All hypothesis tests passed")
else:
    print(f"✗ Hypothesis tests failed")
    print(f"First failing example: {failing_examples[0] if failing_examples else 'N/A'}")

# Test 3: Check with different dtypes
print("\n" + "=" * 60)
print("3. Testing with various dtypes:")

test_data = [1, 0, 2, 0, 3]
sparse = SparseArray(test_data)
print(f"Test data: {test_data}")

test_dtypes = [
    np.float32,
    np.float64,
    np.int32,
    np.int64,
    'float32',
    'float64',
    'int32',
    'int64'
]

for dtype in test_dtypes:
    result = sparse.astype(dtype)
    is_sparse = isinstance(result, SparseArray)
    print(f"  astype({dtype}): {'SparseArray' if is_sparse else type(result).__name__}")

# Test 4: Check with SparseDtype
print("\n" + "=" * 60)
print("4. Testing with SparseDtype:")

from pandas import SparseDtype

sparse = SparseArray([1, 0, 2, 0, 3])
print(f"Test data: [1, 0, 2, 0, 3]")

# Using SparseDtype
sparse_dtype = SparseDtype(np.float64, fill_value=0.0)
result_sparse = sparse.astype(sparse_dtype)
print(f"\nastype(SparseDtype(float64, fill_value=0.0)):")
print(f"  Type: {type(result_sparse)}")
print(f"  Is SparseArray: {isinstance(result_sparse, SparseArray)}")

# Using regular dtype
result_regular = sparse.astype(np.float64)
print(f"\nastype(np.float64):")
print(f"  Type: {type(result_regular)}")
print(f"  Is SparseArray: {isinstance(result_regular, SparseArray)}")

print("\n" + "=" * 60)
print("Summary:")
print("The docstring states 'The output will always be a SparseArray'")
print("But when converting to non-SparseDtype, it returns a numpy array.")
print("This is a contract violation between documentation and implementation.")