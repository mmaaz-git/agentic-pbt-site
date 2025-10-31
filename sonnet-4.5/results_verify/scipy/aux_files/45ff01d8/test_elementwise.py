#!/usr/bin/env python3
"""Test elementwise vs scalar returns"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/scipy_env')

from scipy.integrate import tanhsinh
import numpy as np

print("Testing different return types from integrand function")
print("=" * 60)

# Test with vectorized input
print("\nTest 1: Function that handles array input properly")
def f_vectorized(x):
    # Ensure we return array with same shape as input
    return np.ones_like(x)

try:
    result = tanhsinh(f_vectorized, 0.0, 1.0)
    print(f"Vectorized function: SUCCESS! Result: {result.integral}")
except Exception as e:
    print(f"Vectorized function: FAILED with {type(e).__name__}: {e}")

# Test with scalar return
print("\nTest 2: Function that returns scalar")
def f_scalar(x):
    # Always returns scalar
    return 1.0

try:
    result = tanhsinh(f_scalar, 0.0, 1.0)
    print(f"Scalar function: SUCCESS! Result: {result.integral}")
except Exception as e:
    print(f"Scalar function: FAILED with {type(e).__name__}: {e}")

# Test with explicit array conversion
print("\nTest 3: Function that converts to array")
def f_array(x):
    # Convert to array
    return np.array(1.0)

try:
    result = tanhsinh(f_array, 0.0, 1.0)
    print(f"Array function: SUCCESS! Result: {result.integral}")
except Exception as e:
    print(f"Array function: FAILED with {type(e).__name__}: {e}")

# Test with proper broadcasting
print("\nTest 4: Function that broadcasts correctly")
def f_broadcast(x):
    # Use numpy to ensure proper shape
    result = np.full_like(x, 1.0)
    return result

try:
    result = tanhsinh(f_broadcast, 0.0, 1.0)
    print(f"Broadcast function: SUCCESS! Result: {result.integral}")
except Exception as e:
    print(f"Broadcast function: FAILED with {type(e).__name__}: {e}")

# Let's see what shape is passed in
print("\nTest 5: Investigating input shapes")
def f_investigate(x):
    x_array = np.asarray(x)
    print(f"  Input x shape: {x_array.shape}, ndim: {x_array.ndim}")
    result = 1.0  # Return scalar
    print(f"  Returning: {result}, type: {type(result)}")
    return result

try:
    result = tanhsinh(f_investigate, 0.0, 1.0)
    print(f"Investigation: SUCCESS! Result: {result.integral}")
except Exception as e:
    print(f"Investigation: FAILED with {type(e).__name__}: {e}")

# Now let's try preserve_shape parameter
print("\nTest 6: Using preserve_shape=True with scalar return")
def f_scalar2(x):
    return 1.0

try:
    result = tanhsinh(f_scalar2, 0.0, 1.0, preserve_shape=True)
    print(f"With preserve_shape=True: SUCCESS! Result: {result.integral}")
except Exception as e:
    print(f"With preserve_shape=True: FAILED with {type(e).__name__}: {e}")

# Test infinite bounds (which triggers the problematic code path)
print("\nTest 7: Scalar function with infinite bounds")
def f_scalar_inf(x):
    return np.exp(-x**2)

try:
    result = tanhsinh(f_scalar_inf, -np.inf, np.inf)
    print(f"Scalar with infinite bounds: SUCCESS! Result: {result.integral}")
    print(f"Expected (sqrt(pi)): {np.sqrt(np.pi)}")
except Exception as e:
    print(f"Scalar with infinite bounds: FAILED with {type(e).__name__}: {e}")

# Test a vectorized Gaussian
print("\nTest 8: Vectorized Gaussian with infinite bounds")
def f_gauss_vec(x):
    return np.exp(-x**2) * np.ones_like(x)

try:
    result = tanhsinh(f_gauss_vec, -np.inf, np.inf)
    print(f"Vectorized Gaussian: SUCCESS! Result: {result.integral}")
    print(f"Expected (sqrt(pi)): {np.sqrt(np.pi)}")
except Exception as e:
    print(f"Vectorized Gaussian: FAILED with {type(e).__name__}: {e}")