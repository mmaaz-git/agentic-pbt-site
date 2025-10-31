#!/usr/bin/env python3
"""Test case to reproduce the sizeof return type inconsistency bug"""

import numpy as np
from dask.sizeof import sizeof

# Test 1: Regular array
print("=== Test 1: Regular array ===")
arr_regular = np.ones(100, dtype='f8')
result_regular = sizeof(arr_regular)
print(f"Regular array: type={type(result_regular)}, value={result_regular}")
print(f"Is Python int? {type(result_regular) is int}")
print(f"isinstance(result, int)? {isinstance(result_regular, int)}")

# Test 2: Broadcasted array
print("\n=== Test 2: Broadcasted array ===")
arr_broadcast = np.broadcast_to(1, (100, 100))
result_broadcast = sizeof(arr_broadcast)
print(f"Broadcast array: type={type(result_broadcast)}, value={result_broadcast}")
print(f"Is Python int? {type(result_broadcast) is int}")
print(f"isinstance(result, int)? {isinstance(result_broadcast, int)}")

# Test 3: The assertions from the bug report
print("\n=== Test 3: Assertions ===")
try:
    assert type(result_regular) is int
    print("✓ Regular array returns Python int")
except AssertionError:
    print("✗ Regular array does NOT return Python int")

try:
    assert type(result_broadcast) is int
    print("✓ Broadcast array returns Python int")
except AssertionError:
    print("✗ Broadcast array does NOT return Python int")

# Test 4: The Hypothesis test from the bug report
print("\n=== Test 4: Run property-based test ===")
from hypothesis import given
import hypothesis.strategies as st
import hypothesis.extra.numpy as npst

@given(npst.arrays(dtype=np.float64, shape=st.tuples(st.integers(1, 100))))
def test_sizeof_returns_python_int(arr):
    """sizeof should always return Python int, not numpy integer types"""
    result = sizeof(arr)
    assert isinstance(result, int), f"Expected int, got {type(result)}"
    assert type(result) is int, f"Expected Python int, got {type(result).__name__}"

# Run the hypothesis test
try:
    test_sizeof_returns_python_int()
    print("✓ Hypothesis test passed")
except Exception as e:
    print(f"✗ Hypothesis test failed: {e}")

# Test 5: Check what nbytes actually returns
print("\n=== Test 5: Direct nbytes inspection ===")
xs = arr_broadcast[tuple(slice(None) if s != 0 else slice(1) for s in arr_broadcast.strides)]
print(f"xs.nbytes type: {type(xs.nbytes)}")
print(f"int(xs.nbytes) type: {type(int(xs.nbytes))}")