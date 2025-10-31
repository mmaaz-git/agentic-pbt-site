#!/usr/bin/env python3
"""Demonstrate the bug in Cython.Utils.normalise_float_repr"""

from Cython.Utils import normalise_float_repr

# The failing test case from the bug report
original = 6.103515625e-05
float_str = '6.103515625e-05'
result = normalise_float_repr(float_str)

print(f"Input string:      {float_str}")
print(f"Expected behavior: Normalized string that preserves the float value")
print(f"                   (e.g., .00006103515625 or similar)")
print()
print(f"Actual result:     {result}")
print()
print(f"Original value:    {original}")
print(f"Result value:      {float(result)}")
print(f"Difference:        {abs(original - float(result))}")
print()
print(f"Values are equal?  {float(float_str) == float(result)}")
print()
print("ERROR: Result is off by approximately 10 billion times!")