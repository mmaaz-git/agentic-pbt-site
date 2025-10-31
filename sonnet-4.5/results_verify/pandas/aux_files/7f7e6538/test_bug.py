#!/usr/bin/env python3

import numpy as np
from hypothesis import given, strategies as st
from pandas.core.internals.base import ensure_np_dtype

# Property-based test from the bug report
@given(st.sampled_from([np.dtype(str), np.dtype('U'), np.dtype('U10')]))
def test_ensure_np_dtype_string_to_object(str_dtype):
    result = ensure_np_dtype(str_dtype)
    assert isinstance(result, np.dtype), f"Expected np.dtype, got {type(result)}"
    assert result == np.dtype('object'), f"Expected object dtype, got {result}"

print("Running hypothesis test...")
try:
    test_ensure_np_dtype_string_to_object()
    print("Hypothesis test passed")
except Exception as e:
    print(f"Hypothesis test failed: {e}")

# Direct reproduction test
print("\nDirect reproduction test:")
dtype_var_length = np.dtype(str)
dtype_fixed_length = np.dtype('U10')

result_var = ensure_np_dtype(dtype_var_length)
result_fixed = ensure_np_dtype(dtype_fixed_length)

print(f"Variable-length: {dtype_var_length} -> {result_var}")
print(f"Fixed-length:    {dtype_fixed_length} -> {result_fixed}")

try:
    assert result_var == np.dtype('object')
    print("Variable-length assertion passed")
except AssertionError as e:
    print(f"Variable-length assertion failed: {e}")

try:
    assert result_fixed == np.dtype('object')
    print("Fixed-length assertion passed")
except AssertionError as e:
    print(f"Fixed-length assertion failed: {e}")

# Additional tests
print("\nAdditional tests:")
test_dtypes = [
    np.dtype('U1'),
    np.dtype('U5'),
    np.dtype('U50'),
    np.dtype('U100'),
    np.dtype('U1000'),
]

for dt in test_dtypes:
    result = ensure_np_dtype(dt)
    print(f"  {dt} -> {result} (expected: object)")