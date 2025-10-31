#!/usr/bin/env python3
"""Test script to reproduce the pandas.cut and pandas.qcut bug with small floats"""

import pandas as pd
import numpy as np
import sys

print("Testing pandas version:", pd.__version__)
print()

# Test 1: pd.cut with very small float
print("Test 1: pd.cut with very small float (2.2250738585e-313)")
print("="*60)
try:
    values = [0.0, 0.0, 0.0, 2.2250738585e-313, -1.0]
    print(f"Input values: {values}")
    series = pd.Series(values)
    result = pd.cut(series, bins=2)
    print(f"Result: {result}")
    print("Test 1 PASSED - No exception raised")
except Exception as e:
    print(f"Test 1 FAILED with exception: {type(e).__name__}: {e}")
print()

# Test 2: pd.qcut with very small float
print("Test 2: pd.qcut with very small float (2.2250738585072014e-308)")
print("="*60)
try:
    values = [0.0, 0.0, 0.0, 1.0, 2.2250738585072014e-308]
    print(f"Input values: {values}")
    series = pd.Series(values)
    result = pd.qcut(series, q=3, duplicates='drop')
    print(f"Result: {result}")
    print("Test 2 PASSED - No exception raised")
except Exception as e:
    print(f"Test 2 FAILED with exception: {type(e).__name__}: {e}")
print()

# Test 3: Let's test the _round_frac function directly to understand the issue
print("Test 3: Testing _round_frac behavior with very small numbers")
print("="*60)
from pandas.core.reshape.tile import _round_frac

test_values = [
    2.2250738585e-313,
    2.2250738585072014e-308,
    1e-100,
    1e-50,
    1e-15,
    1e-10,
    0.0
]

for val in test_values:
    try:
        rounded = _round_frac(val, precision=3)
        print(f"_round_frac({val:.3e}, 3) = {rounded}")
        if not np.isfinite(rounded):
            print(f"  WARNING: Result is not finite! (NaN or Inf)")
    except Exception as e:
        print(f"_round_frac({val:.3e}, 3) raised {type(e).__name__}: {e}")
print()

# Test 4: Run the hypothesis test
print("Test 4: Running hypothesis test")
print("="*60)
from hypothesis import given, strategies as st, settings

@given(
    values=st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
                    min_size=5, max_size=100),
    bins=st.integers(min_value=2, max_value=10)
)
@settings(max_examples=200)
def test_cut_result_length(values, bins):
    series = pd.Series(values)
    result = pd.cut(series, bins=bins)
    assert len(result) == len(series)

try:
    test_cut_result_length()
    print("Hypothesis test completed without finding issues")
except AssertionError as e:
    print(f"Hypothesis test found assertion error: {e}")
except Exception as e:
    print(f"Hypothesis test failed with: {type(e).__name__}: {e}")