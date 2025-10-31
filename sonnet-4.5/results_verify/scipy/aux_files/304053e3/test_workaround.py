#!/usr/bin/env python3
"""Test the workaround mentioned in the bug report"""

from scipy import integrate
import numpy as np

print("Testing workaround: integrate.tanhsinh(lambda x: np.full_like(x, 0.0), 0.0, 1.0)")
try:
    result = integrate.tanhsinh(lambda x: np.full_like(x, 0.0), 0.0, 1.0)
    print(f"Success with workaround: {result}")
    print(f"Result integral: {result.integral}")
except Exception as e:
    print(f"Error even with workaround: {type(e).__name__}: {e}")

# Also test with other scipy functions
print("\nTesting with other scipy integration functions:")

print("integrate.quad(lambda x: 0.0, 0.0, 1.0):")
try:
    result = integrate.quad(lambda x: 0.0, 0.0, 1.0)
    print(f"  Success: {result}")
except Exception as e:
    print(f"  Error: {type(e).__name__}: {e}")

print("\nintegrate.quad_vec(lambda x: 0.0, 0.0, 1.0):")
try:
    result = integrate.quad_vec(lambda x: 0.0, 0.0, 1.0)
    print(f"  Success: {result}")
except Exception as e:
    print(f"  Error: {type(e).__name__}: {e}")