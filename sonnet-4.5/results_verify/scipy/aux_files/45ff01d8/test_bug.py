#!/usr/bin/env python3
"""Test the reported bug with scipy.integrate.tanhsinh"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/scipy_env')

from scipy.integrate import tanhsinh
import numpy as np

print("Testing scipy.integrate.tanhsinh with constant functions")
print("=" * 60)

# Test 1: Simple constant function returning 1.0
print("\nTest 1: Constant function f(x) = 1.0 on [0, 1]")
def f1(x):
    return 1.0

try:
    result = tanhsinh(f1, 0.0, 1.0)
    print(f"Success! Result: {result.integral}")
    print(f"Expected: 1.0")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Constant function returning 5.0
print("\nTest 2: Constant function f(x) = 5.0 on [-1, 1]")
def f2(x):
    return 5.0

try:
    result = tanhsinh(f2, -1.0, 1.0)
    print(f"Success! Result: {result.integral}")
    print(f"Expected: 10.0")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Test 3: Constant function returning 0.0
print("\nTest 3: Constant function f(x) = 0.0 on [0, 1]")
def f3(x):
    return 0.0

try:
    result = tanhsinh(f3, 0.0, 1.0)
    print(f"Success! Result: {result.integral}")
    print(f"Expected: 0.0")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Test 4: Non-constant linear function for comparison
print("\nTest 4: Linear function f(x) = x on [0, 1]")
def f4(x):
    return x

try:
    result = tanhsinh(f4, 0.0, 1.0)
    print(f"Success! Result: {result.integral}")
    print(f"Expected: 0.5")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Test 5: Quadratic function for comparison
print("\nTest 5: Quadratic function f(x) = x**2 on [0, 1]")
def f5(x):
    return x**2

try:
    result = tanhsinh(f5, 0.0, 1.0)
    print(f"Success! Result: {result.integral}")
    print(f"Expected: 0.333...")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Test 6: Let's check what the function actually returns
print("\nTest 6: Checking what constant function returns")
def f_const(x):
    result = 1.0
    print(f"  f({x}) = {result}, type: {type(result)}, shape: {np.array(result).shape}")
    return result

try:
    print("Calling tanhsinh with verbose function:")
    result = tanhsinh(f_const, 0.0, 1.0)
    print(f"Success! Result: {result.integral}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")