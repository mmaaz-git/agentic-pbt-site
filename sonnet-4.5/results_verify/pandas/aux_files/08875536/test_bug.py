#!/usr/bin/env python3

# Test 1: Basic reproduction
from pandas.api.types import pandas_dtype
from pandas.core.dtypes.common import _get_dtype
import numpy as np

print("=== Test 1: Basic reproduction ===")
result = pandas_dtype(None)
print(f"pandas_dtype(None) = {result}")
print(f"Type: {type(result)}")

# Check numpy behavior
print(f"\nnumpy.dtype(None) = {np.dtype(None)}")

# Test 2: Check _get_dtype
print("\n=== Test 2: _get_dtype behavior ===")
try:
    result = _get_dtype(None)
    print(f"_get_dtype(None) = {result}")
except TypeError as e:
    print(f"_get_dtype(None) raised TypeError: {e}")

# Test 3: Run the hypothesis test
print("\n=== Test 3: Hypothesis test ===")
from hypothesis import given, strategies as st

@given(st.none())
def test_pandas_dtype_none_should_raise_typeerror(value):
    try:
        result = pandas_dtype(value)
        print(f"Test FAILED: pandas_dtype(None) returned {result}, expected TypeError")
        return False
    except TypeError:
        print(f"Test PASSED: pandas_dtype(None) raised TypeError as expected")
        return True

# Run without calling directly - hypothesis runs it
def manual_test():
    try:
        result = pandas_dtype(None)
        print(f"Test FAILED: pandas_dtype(None) returned {result}, expected TypeError")
    except TypeError:
        print(f"Test PASSED: pandas_dtype(None) raised TypeError as expected")

manual_test()

# Test 4: Check various related dtype functions
print("\n=== Test 4: Related dtype functions ===")
from pandas.api.types import is_dtype_equal

print(f"is_dtype_equal(None, None) = {is_dtype_equal(None, None)}")
print(f"is_dtype_equal(None, 'float64') = {is_dtype_equal(None, 'float64')}")

# Test 5: Check docstring
print("\n=== Test 5: pandas_dtype docstring ===")
print(pandas_dtype.__doc__)