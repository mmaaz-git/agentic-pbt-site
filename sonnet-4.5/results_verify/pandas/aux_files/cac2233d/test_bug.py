#!/usr/bin/env python3
"""Test script to reproduce the reported bug in zsqrt"""

import sys
import traceback
import numpy as np
import pandas as pd
from pandas.core.window.common import zsqrt

print("Testing pandas.core.window.common.zsqrt")
print("=" * 50)

# Test 1: Scalar input (reported bug)
print("\nTest 1: Scalar input zsqrt(0.0)")
try:
    result = zsqrt(0.0)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    traceback.print_exc()

# Test 2: Scalar negative input
print("\nTest 2: Scalar negative input zsqrt(-4.0)")
try:
    result = zsqrt(-4.0)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Test 3: Scalar positive input
print("\nTest 3: Scalar positive input zsqrt(4.0)")
try:
    result = zsqrt(4.0)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Test 4: NumPy array input
print("\nTest 4: NumPy array input")
try:
    arr = np.array([1.0, 4.0, -1.0, 9.0])
    result = zsqrt(arr)
    print(f"Input: {arr}")
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Test 5: Pandas Series input
print("\nTest 5: Pandas Series input")
try:
    series = pd.Series([1.0, 4.0, -1.0, 9.0])
    result = zsqrt(series)
    print(f"Input: {series.values}")
    print(f"Result: {result.values}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Test 6: Pandas DataFrame input
print("\nTest 6: Pandas DataFrame input")
try:
    df = pd.DataFrame({'A': [1.0, 4.0, -1.0, 9.0], 'B': [16.0, -4.0, 25.0, 36.0]})
    result = zsqrt(df)
    print(f"Input:\n{df}")
    print(f"Result:\n{result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Test 7: Check how it's actually used in ewm.py context
print("\nTest 7: Usage context simulation (like in ewm.py:892)")
try:
    # Simulate x_var * y_var producing a scalar
    x_var = 4.0
    y_var = 9.0
    product = x_var * y_var
    print(f"x_var * y_var = {product} (type: {type(product)})")
    result = zsqrt(product)
    print(f"zsqrt(x_var * y_var) = {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    traceback.print_exc()

# Run the hypothesis test
print("\n" + "=" * 50)
print("Running Hypothesis Property-Based Test")
print("=" * 50)
try:
    from hypothesis import given, strategies as st

    @given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10))
    def test_zsqrt_always_nonnegative(x):
        result = zsqrt(x)
        assert result >= 0, f"zsqrt({x}) = {result} is negative"

    # Run the test
    test_zsqrt_always_nonnegative()
    print("Hypothesis test passed!")
except Exception as e:
    print(f"Hypothesis test failed: {type(e).__name__}: {e}")
    traceback.print_exc()