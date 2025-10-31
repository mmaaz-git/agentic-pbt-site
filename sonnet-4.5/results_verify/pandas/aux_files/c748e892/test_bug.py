#!/usr/bin/env python3
"""Reproduce the reported bug in normalise_float_repr"""

from Cython.Utils import normalise_float_repr

# Test case from bug report
test_input = '1.192092896e-07'
result = normalise_float_repr(test_input)

print(f"Input:    {test_input}")
print(f"Output:   {result}")
print(f"Expected: .0000001192092896")
print()
print(f"Input float value:  {float(test_input)}")
print(f"Output float value: {float(result)}")

# Check if they're equal
try:
    assert float(test_input) == float(result)
    print("Assertion passed: Values are equal")
except AssertionError:
    print("AssertionError: Values are NOT equal")
    print(f"Difference: {float(result) - float(test_input)}")