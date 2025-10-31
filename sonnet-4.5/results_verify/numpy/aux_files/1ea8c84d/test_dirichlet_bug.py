#!/usr/bin/env python3
"""Test script to reproduce the numpy.random.dirichlet bug with zero alpha parameters."""

import numpy as np
import numpy.random as npr
from hypothesis import given, strategies as st, assume
import sys

print("NumPy version:", np.__version__)
print("-" * 60)

# Test 1: Property-based test from bug report
print("Test 1: Property-based test with hypothesis")
print("-" * 40)

try:
    @given(st.integers(min_value=2, max_value=10))
    def test_dirichlet_all_zeros_violates_simplex_constraint(size):
        rng = npr.default_rng(42)

        alpha = np.zeros(size)
        result = rng.dirichlet(alpha)

        assert np.isclose(result.sum(), 1.0), f"Sum is {result.sum()}, expected 1.0"

    # Run the test
    test_dirichlet_all_zeros_violates_simplex_constraint()
    print("Property test PASSED (no error with zero alphas)")
except AssertionError as e:
    print(f"Property test FAILED with assertion error: {e}")
except ValueError as e:
    print(f"Property test raised ValueError (as it should): {e}")
except Exception as e:
    print(f"Property test raised unexpected error: {type(e).__name__}: {e}")

print()

# Test 2: Direct reproduction from bug report
print("Test 2: Direct reproduction with zero alphas")
print("-" * 40)

try:
    rng = npr.default_rng(42)

    alpha_zeros = [0.0, 0.0, 0.0]
    result = rng.dirichlet(alpha_zeros)

    print(f"Alpha: {alpha_zeros}")
    print(f"Result: {result}")
    print(f"Sum: {result.sum()}")
    print(f"Is sum close to 1.0? {np.isclose(result.sum(), 1.0)}")
except ValueError as e:
    print(f"ValueError raised (as expected per documentation): {e}")
except Exception as e:
    print(f"Unexpected error: {type(e).__name__}: {e}")

print()

# Test 3: Test with negative alpha values
print("Test 3: Test with negative alpha values")
print("-" * 40)

try:
    rng = npr.default_rng(42)

    alpha_negative = [-1.0, 2.0, 3.0]
    result = rng.dirichlet(alpha_negative)

    print(f"Alpha: {alpha_negative}")
    print(f"Result: {result}")
    print(f"Sum: {result.sum()}")
except ValueError as e:
    print(f"ValueError raised (as expected): {e}")
except Exception as e:
    print(f"Unexpected error: {type(e).__name__}: {e}")

print()

# Test 4: Test with mixed zero and positive values
print("Test 4: Test with mixed zero and positive alpha values")
print("-" * 40)

try:
    rng = npr.default_rng(42)

    alpha_mixed = [0.0, 1.0, 2.0]
    result = rng.dirichlet(alpha_mixed)

    print(f"Alpha: {alpha_mixed}")
    print(f"Result: {result}")
    print(f"Sum: {result.sum()}")
    print(f"Is sum close to 1.0? {np.isclose(result.sum(), 1.0)}")
except ValueError as e:
    print(f"ValueError raised: {e}")
except Exception as e:
    print(f"Unexpected error: {type(e).__name__}: {e}")

print()

# Test 5: Test with very small positive values (should work)
print("Test 5: Test with very small positive alpha values")
print("-" * 40)

try:
    rng = npr.default_rng(42)

    alpha_small = [0.001, 0.001, 0.001]
    result = rng.dirichlet(alpha_small)

    print(f"Alpha: {alpha_small}")
    print(f"Result: {result}")
    print(f"Sum: {result.sum()}")
    print(f"Is sum close to 1.0? {np.isclose(result.sum(), 1.0)}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print()

# Test 6: Compare with beta distribution behavior
print("Test 6: Compare with beta distribution (which should reject zero)")
print("-" * 40)

try:
    rng = npr.default_rng(42)

    # Beta with zero parameters
    result = rng.beta(0.0, 1.0)
    print(f"Beta(0, 1) result: {result}")
except ValueError as e:
    print(f"Beta correctly raises ValueError for zero parameter: {e}")
except Exception as e:
    print(f"Unexpected error from beta: {type(e).__name__}: {e}")

print()

# Test 7: Verify mathematical property - sum should always be 1
print("Test 7: Verify sum=1 property with valid inputs")
print("-" * 40)

try:
    rng = npr.default_rng(42)

    for alpha in [[1, 1, 1], [0.5, 0.5, 0.5], [2, 3, 4], [10, 20, 30]]:
        result = rng.dirichlet(alpha)
        print(f"Alpha: {alpha}, Sum: {result.sum():.15f}, Close to 1? {np.isclose(result.sum(), 1.0)}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")