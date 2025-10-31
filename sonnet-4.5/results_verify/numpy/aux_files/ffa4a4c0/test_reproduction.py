#!/usr/bin/env python3
"""Test reproduction for numpy.matrix.I documentation bug"""

import numpy as np
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
from hypothesis import strategies as st

# First, reproduce the exact failing example from the bug report
print("=" * 60)
print("Test 1: Reproducing the exact example from the bug report")
print("=" * 60)

m = np.matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

inv = m.I
product1 = inv * m
product2 = m * inv
eye_from_docstring = np.matrix(np.eye(m[0,:].size))

print(f"Matrix shape: {m.shape}")
print(f"Pseudoinverse shape: {inv.shape}")
print(f"inv * m shape: {product1.shape}")
print(f"m * inv shape: {product2.shape}")
print(f"eye(m[0,:].size) shape: {eye_from_docstring.shape}")

# Check if we can compare them
print("\nTrying to compare product2 with eye_from_docstring:")
try:
    result = (product2 == eye_from_docstring)
    print(f"Comparison successful: {result}")
except ValueError as e:
    print(f"ValueError: {e}")

# Now check what the products actually look like
print("\nProduct1 (inv * m):")
print(product1)
print("\nProduct2 (m * inv):")
print(product2)
print("\nExpected eye matrix from docstring (eye(3)):")
print(eye_from_docstring)

# Verify mathematical properties of pseudoinverse
print("\n" + "=" * 60)
print("Test 2: Verifying mathematical properties of pseudoinverse")
print("=" * 60)

# For a non-square matrix, the pseudoinverse satisfies different properties
# Check: A * A+ * A ≈ A
check1 = m * inv * m
print(f"\nm * inv * m (should equal m):")
print(check1)
print(f"Original m:")
print(m)
print(f"Are they close? {np.allclose(check1, m)}")

# Check: A+ * A * A+ ≈ A+
check2 = inv * m * inv
print(f"\ninv * m * inv (should equal inv):")
print(check2)
print(f"Original inv:")
print(inv)
print(f"Are they close? {np.allclose(check2, inv)}")

# Test with the zero matrix from hypothesis test
print("\n" + "=" * 60)
print("Test 3: Testing with zero matrix from hypothesis")
print("=" * 60)

zero_m = np.matrix([[0., 0., 0.], [0., 0., 0.]])
print(f"Zero matrix:\n{zero_m}")
try:
    zero_inv = zero_m.I
    print(f"Pseudoinverse of zero matrix:\n{zero_inv}")

    product1_zero = zero_inv * zero_m
    product2_zero = zero_m * zero_inv
    eye_zero = np.matrix(np.eye(zero_m[0,:].size))

    print(f"\nzero_inv * zero_m:\n{product1_zero}")
    print(f"zero_m * zero_inv:\n{product2_zero}")
    print(f"Expected eye(3):\n{eye_zero}")

    print(f"\nAre they equal to eye? {np.allclose(product1_zero, eye_zero)} and {np.allclose(product2_zero, eye_zero)}")
except Exception as e:
    print(f"Error computing pseudoinverse: {e}")

# Test with a square non-singular matrix (should work according to docstring)
print("\n" + "=" * 60)
print("Test 4: Testing with square non-singular matrix")
print("=" * 60)

square_m = np.matrix([[1, 2], [3, 4]])
print(f"Square matrix:\n{square_m}")
square_inv = square_m.I
print(f"Inverse:\n{square_inv}")

prod1_sq = square_inv * square_m
prod2_sq = square_m * square_inv
eye_sq = np.matrix(np.eye(square_m[0,:].size))

print(f"\nsquare_inv * square_m:\n{prod1_sq}")
print(f"square_m * square_inv:\n{prod2_sq}")
print(f"Expected eye(2):\n{eye_sq}")

print(f"\nAre they equal to eye?")
print(f"  inv * m == eye: {np.allclose(prod1_sq, eye_sq)}")
print(f"  m * inv == eye: {np.allclose(prod2_sq, eye_sq)}")

# Now check the source code behavior
print("\n" + "=" * 60)
print("Test 5: Understanding the implementation")
print("=" * 60)

# Looking at the source code: when M != N, it uses pinv, not inv
test_nonsquare = np.matrix([[1, 2, 3], [4, 5, 6]])
M, N = test_nonsquare.shape
print(f"For non-square matrix with shape {test_nonsquare.shape}:")
print(f"  M={M}, N={N}, M==N: {M==N}")
print(f"  Implementation uses: {'inv' if M==N else 'pinv'}")

# Directly compare with numpy.linalg.pinv
from numpy.linalg import pinv
direct_pinv = np.matrix(pinv(test_nonsquare))
property_I = test_nonsquare.I
print(f"\nDirect pinv result:\n{direct_pinv}")
print(f"Property .I result:\n{property_I}")
print(f"Are they the same? {np.allclose(direct_pinv, property_I)}")