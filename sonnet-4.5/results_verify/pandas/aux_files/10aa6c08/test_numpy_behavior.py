#!/usr/bin/env python3
"""Test to understand numpy's behavior with None and why it returns float64"""

import numpy as np

print("Understanding NumPy's dtype(None) behavior:")
print("=" * 60)

# Test np.dtype(None)
result = np.dtype(None)
print(f"np.dtype(None) = {result}")
print(f"Type: {type(result)}")

# Test if this is documented numpy behavior
print("\nChecking if this is standard numpy behavior:")
print(f"np.dtype(None) == np.dtype('float64'): {result == np.dtype('float64')}")
print(f"np.dtype(None) is np.dtype('float64'): {result is np.dtype('float64')}")

# Check numpy version
print(f"\nNumPy version: {np.__version__}")

# Test other "empty" values
print("\nTesting other 'empty' values with np.dtype:")
test_values = [
    (None, "None"),
    (type(None), "type(None)"),
    ("None", "'None'"),
]

for val, desc in test_values:
    try:
        result = np.dtype(val)
        print(f"  {desc:15} -> {result}")
    except Exception as e:
        print(f"  {desc:15} -> {type(e).__name__}: {e}")

# Check if there's any numpy documentation about this
print("\nNote: NumPy's np.dtype(None) returning float64 appears to be")
print("an undocumented quirk/legacy behavior of NumPy.")