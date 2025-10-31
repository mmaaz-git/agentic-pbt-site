#!/usr/bin/env python3
"""Compare behavior of different scipy integration functions with scalar returns"""

from scipy import integrate
import numpy as np

# Test functions
test_functions = [
    ("constant", lambda x: 0.0, 0.0, 1.0),
    ("linear", lambda x: x, 0.0, 1.0),
    ("quadratic", lambda x: x**2, 0.0, 1.0),
]

print("Testing different scipy integration functions with scalar-returning functions:\n")

for func_name, func, a, b in test_functions:
    print(f"Function: {func_name} from {a} to {b}")
    print("-" * 50)

    # Test quad
    print("  quad: ", end="")
    try:
        result = integrate.quad(func, a, b)
        print(f"Success - Result: {result[0]:.6f}")
    except Exception as e:
        print(f"Failed - {type(e).__name__}: {e}")

    # Test quad_vec
    print("  quad_vec: ", end="")
    try:
        result = integrate.quad_vec(func, a, b)
        print(f"Success - Result: {result[0]:.6f}")
    except Exception as e:
        print(f"Failed - {type(e).__name__}: {e}")

    # Test simpson
    print("  simpson: ", end="")
    try:
        x = np.linspace(a, b, 100)
        y = [func(xi) for xi in x]
        result = integrate.simpson(y, x=x)
        print(f"Success - Result: {result:.6f}")
    except Exception as e:
        print(f"Failed - {type(e).__name__}: {e}")

    # Test tanhsinh
    print("  tanhsinh: ", end="")
    try:
        result = integrate.tanhsinh(func, a, b)
        print(f"Success - Result: {result.integral:.6f}")
    except Exception as e:
        print(f"Failed - {type(e).__name__}")

    print()

# Now test with array-returning functions (the "correct" way for tanhsinh)
print("\nTesting tanhsinh with array-returning functions (workaround):")
print("-" * 50)

for func_name, scalar_func, a, b in test_functions:
    # Create array version
    array_func = lambda x: np.full_like(x, scalar_func(x) if np.isscalar(x) else scalar_func(x[0]), dtype=float)

    print(f"Function: {func_name} (array version) from {a} to {b}")
    try:
        result = integrate.tanhsinh(array_func, a, b)
        print(f"  Success - Result: {result.integral:.6f}, Success: {result.success}")
    except Exception as e:
        print(f"  Failed - {type(e).__name__}: {e}")

# Test with proper vectorized function
print("\nTesting tanhsinh with properly vectorized functions:")
print("-" * 50)

vectorized_functions = [
    ("constant", lambda x: np.zeros_like(x), 0.0, 1.0),
    ("linear", lambda x: x, 0.0, 1.0),  # This already works with arrays
    ("quadratic", lambda x: x**2, 0.0, 1.0),  # This also works
]

for func_name, func, a, b in vectorized_functions:
    print(f"Function: {func_name} from {a} to {b}")
    try:
        result = integrate.tanhsinh(func, a, b)
        print(f"  Success - Result: {result.integral:.6f}, Success: {result.success}")
    except Exception as e:
        print(f"  Failed - {type(e).__name__}: {e}")