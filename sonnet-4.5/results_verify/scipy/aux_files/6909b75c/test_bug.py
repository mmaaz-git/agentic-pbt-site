#!/usr/bin/env python3
"""Test the reported bug in scipy.stats.quantile"""

import numpy as np
from scipy import stats

print("Testing scipy.stats.quantile with integer p values\n")

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

print("Test array:", x)
print()

# Test with integer 0
print("Test 1: stats.quantile(x, 0) - integer 0")
try:
    result = stats.quantile(x, 0)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
print()

# Test with float 0.0
print("Test 2: stats.quantile(x, 0.0) - float 0.0")
try:
    result = stats.quantile(x, 0.0)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
print()

# Test with integer 1
print("Test 3: stats.quantile(x, 1) - integer 1")
try:
    result = stats.quantile(x, 1)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
print()

# Test with float 1.0
print("Test 4: stats.quantile(x, 1.0) - float 1.0")
try:
    result = stats.quantile(x, 1.0)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
print()

# Compare with numpy.percentile
print("Comparison with numpy.percentile:")
print(f"np.percentile(x, 0): {np.percentile(x, 0)}")
print(f"np.percentile(x, 100): {np.percentile(x, 100)}")