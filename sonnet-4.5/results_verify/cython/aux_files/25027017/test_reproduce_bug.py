#!/usr/bin/env python3
"""Reproduce the reported bug in normalise_float_repr"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from Cython.Utils import normalise_float_repr
import math

print("Testing with Hypothesis...")
print("-" * 50)

@settings(max_examples=1000)
@given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e100, max_value=1e100))
def test_normalise_float_repr_value_preservation(x):
    float_str = str(x)
    normalized = normalise_float_repr(float_str)
    try:
        assert math.isclose(float(normalized), float(float_str), rel_tol=1e-15)
    except ValueError as e:
        print(f"ValueError with input: {float_str}")
        print(f"Normalized output: {normalized}")
        print(f"Error: {e}")
        raise

# Run the test
try:
    test_normalise_float_repr_value_preservation()
    print("Hypothesis test passed (no issues found in 1000 examples)")
except Exception as e:
    print(f"Hypothesis test failed: {e}")

print("\n" + "=" * 50)
print("Testing specific failing input from bug report...")
print("-" * 50)

x = "-3.833509682449162e-128"
result = normalise_float_repr(x)
print(f"Input:  {x}")
print(f"Output: {result}")

try:
    converted = float(result)
    print(f"Successfully converted back to float: {converted}")
except ValueError as e:
    print(f"ValueError: {e}")
    print("The output cannot be converted back to a float!")

# Test a few more negative numbers with exponents
print("\n" + "=" * 50)
print("Testing additional negative numbers with exponents...")
print("-" * 50)

test_cases = [
    "-1e-10",
    "-2.5e-50",
    "-9.9e-100",
    "-1.23456e-200",
    "-5e10",
    "-3.14e5"
]

for test in test_cases:
    result = normalise_float_repr(test)
    print(f"Input:  {test:20s} Output: {result}")
    try:
        float(result)
        print(f"  -> Valid float")
    except ValueError:
        print(f"  -> INVALID - Cannot convert back to float!")