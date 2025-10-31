#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env')

from Cython.Utils import normalise_float_repr

# Test case 1: 5.960464477539063e-08
print("=== Test case 1 ===")
f1 = 5.960464477539063e-08
float_str1 = str(f1)
result1 = normalise_float_repr(float_str1)
print(f"Input value: {f1}")
print(f"Input string: {float_str1}")
print(f"Result: {result1}")
try:
    parsed1 = float(result1)
    print(f"Float of result: {parsed1}")
    print(f"Values match: {f1 == parsed1}")
    print(f"Expected: {f1}, Got: {parsed1}")
except ValueError as e:
    print(f"ERROR: Cannot parse result as float: {e}")

print()

# Test case 2: -3.0929648190816446e-178
print("=== Test case 2 ===")
f2 = -3.0929648190816446e-178
float_str2 = str(f2)
result2 = normalise_float_repr(float_str2)
print(f"Input value: {f2}")
print(f"Input string: {float_str2}")
print(f"Result: {result2}")
try:
    parsed2 = float(result2)
    print(f"Float of result: {parsed2}")
    print(f"Values match: {f2 == parsed2}")
    print(f"Expected: {f2}, Got: {parsed2}")
except ValueError as e:
    print(f"ERROR: Cannot parse result as float: {e}")