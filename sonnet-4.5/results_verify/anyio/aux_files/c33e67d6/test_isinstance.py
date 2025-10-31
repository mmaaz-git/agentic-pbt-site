#!/usr/bin/env python3
"""Test isinstance behavior with floats and ints"""

import math

test_values = [0, 0.0, 1, 1.0, 5, 5.0, 5.5, 3.14, math.inf]

print("Testing isinstance(value, int) behavior:")
print("=" * 50)

for val in test_values:
    is_int = isinstance(val, int)
    is_float = isinstance(val, float)
    type_str = type(val).__name__

    print(f"Value: {val:6} | Type: {type_str:5} | isinstance(int): {is_int:5} | isinstance(float): {is_float:5}")

print("\n" + "=" * 50)
print("\nTesting the validation logic from anyio:")
print("if max_buffer_size != math.inf and not isinstance(max_buffer_size, int):")
print()

for val in test_values:
    if val != math.inf and not isinstance(val, int):
        print(f"  {val:6} would be REJECTED (raises ValueError)")
    else:
        print(f"  {val:6} would be ACCEPTED")