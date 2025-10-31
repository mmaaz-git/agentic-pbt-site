#!/usr/bin/env python3
import numpy as np
from xarray.compat.npcompat import isdtype

print(f"NumPy version: {np.__version__}")
print(f"Has native isdtype: {hasattr(np, 'isdtype')}")

scalar = np.int32(5)
print(f"Testing with scalar: {scalar}")
print(f"Type of scalar: {type(scalar)}")
print(f"Is np.generic: {isinstance(scalar, np.generic)}")

try:
    result = isdtype(scalar, 'integral')
    print(f"Success: {result}")
    print("This works on NumPy < 2.0 or with fallback implementation")
except TypeError as e:
    print(f"TypeError: {e}")
    print("This fails on NumPy >= 2.0")

# Test with dtype instead
print(f"\nTesting with dtype: {scalar.dtype}")
try:
    result = isdtype(scalar.dtype, 'integral')
    print(f"Success: {result}")
    print("This works on all NumPy versions")
except Exception as e:
    print(f"Error: {e}")

# Test with type
print(f"\nTesting with type: {type(scalar)}")
try:
    result = isdtype(type(scalar), 'integral')
    print(f"Success: {result}")
except Exception as e:
    print(f"Error: {e}")