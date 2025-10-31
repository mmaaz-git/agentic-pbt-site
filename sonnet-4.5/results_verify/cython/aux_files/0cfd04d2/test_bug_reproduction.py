#!/usr/bin/env python3
"""Test reproduction for normalise_float_repr bug"""

from hypothesis import given, strategies as st
import math
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')
from Cython.Utils import normalise_float_repr

# First, test with the specific examples provided
print("=== Testing specific examples from bug report ===")
print()

print("Bug 1: Invalid float string for very small negative number")
f1 = -1.670758163823954e-133
float_str1 = str(f1)
result1 = normalise_float_repr(float_str1)
print(f"Input: {float_str1}")
print(f"Output: {result1}")
try:
    converted = float(result1)
    print(f"Converted back: {converted}")
except ValueError as e:
    print(f"ERROR: Cannot convert back to float: {e}")
print()

print("Bug 2: Value corruption for small numbers")
f2 = 1.114036198514633e-05
float_str2 = str(f2)
result2 = normalise_float_repr(float_str2)
print(f"Input: {float_str2} = {f2}")
print(f"Output: {result2}")
try:
    converted2 = float(result2)
    print(f"Converted: {converted2}")
    print(f"Error: {abs(converted2 - f2) / abs(f2) * 100:.1f}%")
except ValueError as e:
    print(f"ERROR: Cannot convert back to float: {e}")
print()

# Now run the hypothesis test
print("=== Running hypothesis test ===")

@given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e100, max_value=1e100))
def test_normalise_float_repr_preserves_value(f):
    float_str = str(f)
    result = normalise_float_repr(float_str)
    try:
        converted = float(result)
        assert math.isclose(converted, float(float_str), rel_tol=1e-14)
    except ValueError:
        # If we can't convert it back to float, the function is broken
        print(f"Failed to convert result back to float: input={float_str}, output={result}")
        raise

# Run a limited number of hypothesis tests
from hypothesis import settings
with settings(max_examples=20):
    try:
        test_normalise_float_repr_preserves_value()
        print("Hypothesis test passed for 20 examples")
    except Exception as e:
        print(f"Hypothesis test failed: {e}")