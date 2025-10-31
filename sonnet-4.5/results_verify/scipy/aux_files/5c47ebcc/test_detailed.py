#!/usr/bin/env python3
"""Detailed investigation of when tanhsinh fails"""

from scipy import integrate
import numpy as np

print("Testing various constant functions with tanhsinh:\n")

# Test different constant values
constants = [0.0, 1.0, -1.0, 5.0, 0.5]

for c in constants:
    print(f"Testing lambda x: {c}")
    try:
        result = integrate.tanhsinh(lambda x: c, 0.0, 1.0)
        print(f"  Success: {result.integral}")
    except IndexError:
        print(f"  Failed: IndexError")
    except Exception as e:
        print(f"  Failed: {type(e).__name__}: {e}")

print("\nTesting functions that use x but return same type:")

test_cases = [
    ("lambda x: 0.0", lambda x: 0.0),
    ("lambda x: 1.0", lambda x: 1.0),
    ("lambda x: 0*x", lambda x: 0*x),
    ("lambda x: 0*x + 1", lambda x: 0*x + 1),
    ("lambda x: x*0", lambda x: x*0),
    ("lambda x: x - x", lambda x: x - x),
    ("lambda x: x/x - 1", lambda x: x/x - 1),  # Returns 0
]

for desc, func in test_cases:
    print(f"\nTesting {desc}")
    try:
        result = integrate.tanhsinh(func, 0.1, 1.0)  # Start from 0.1 to avoid division by zero
        print(f"  Success: {result.integral:.6f}")
    except IndexError:
        print(f"  Failed: IndexError")
    except Exception as e:
        print(f"  Failed: {type(e).__name__}: {e}")

print("\n\nChecking what tanhsinh passes to the function:")

def debug_func(x):
    print(f"  Called with x: type={type(x)}, shape={x.shape if hasattr(x, 'shape') else 'N/A'}, value={x if np.size(x) <= 5 else f'{x[:5]}...'}")
    return x * 0  # This should work since it preserves array nature

print("\nCalling tanhsinh with debug function (lambda x: x*0):")
try:
    result = integrate.tanhsinh(debug_func, 0.0, 1.0)
    print(f"  Final result: {result.integral}")
except Exception as e:
    print(f"  Failed: {type(e).__name__}: {e}")

def debug_func_scalar(x):
    print(f"  Called with x: type={type(x)}, shape={x.shape if hasattr(x, 'shape') else 'N/A'}, value={x if np.size(x) <= 5 else f'{x[:5]}...'}")
    return 0.0  # Returns scalar

print("\nCalling tanhsinh with debug function (lambda x: 0.0):")
try:
    result = integrate.tanhsinh(debug_func_scalar, 0.0, 1.0)
    print(f"  Final result: {result.integral}")
except Exception as e:
    print(f"  Failed: {type(e).__name__}")