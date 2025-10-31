#!/usr/bin/env python3
"""Test script to understand float values."""

import numpy as np
import sys

print("Float information:")
print(f"sys.float_info.min: {sys.float_info.min}")
print(f"numpy.finfo(float).min: {np.finfo(float).min}")
print(f"numpy.finfo(float).tiny: {np.finfo(float).tiny}")
print(f"numpy.finfo(float).eps: {np.finfo(float).eps}")

# Test value used in bug report
test_val = 5e-324
print(f"\nTest value 5e-324: {test_val}")
print(f"Is it > 0? {test_val > 0}")
print(f"Is it == 0? {test_val == 0}")
print(f"Type: {type(test_val)}")

# Python's smallest positive float
print(f"\nSmallest positive float in Python: {sys.float_info.min}")
print(f"Smallest subnormal: {5e-324}")

# Test numpy operations
print(f"\nnp.ceil(5e-324): {np.ceil(5e-324)}")
print(f"np.ceil(0.0): {np.ceil(0.0)}")
print(f"int(np.ceil(5e-324)): {int(np.ceil(5e-324))}")
print(f"int(np.ceil(0.0)): {int(np.ceil(0.0))}")

# Test what happens with division
print("\nDivision tests:")
print(f"1.0 / 5e-324 = {1.0 / 5e-324}")
print(f"5e-324 / 1.0 = {5e-324 / 1.0}")
print(f"0.1 / np.inf = {0.1 / np.inf}")
print(f"np.ceil(0.1 / np.inf) = {np.ceil(0.1 / np.inf)}")