#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/scipy_env')

import numpy as np
from scipy.interpolate import lagrange

# Test case from the bug report
x = np.array([0.0, 1.0, 260158.0])
y = np.array([32156.0, 0.0, 0.0])

poly = lagrange(x, y)

print("Verifying interpolation property:")
for i in range(len(x)):
    result = poly(x[i])
    expected = y[i]
    error = abs(result - expected)
    print(f"poly({x[i]}) = {result}, expected {expected}, error = {error}")

# Also let's check the polynomial coefficients
print(f"\nPolynomial coefficients: {poly.coef}")
print(f"Polynomial degree: {len(poly.coef) - 1}")

# Let's verify manually what the polynomial should be
# For 3 points, we get a degree 2 polynomial
# The polynomial through (0, 32156), (1, 0), (260158, 0) should be:
# We can verify this using the Lagrange formula directly

def lagrange_manual(x_points, y_points, x_eval):
    """Manual Lagrange interpolation for comparison"""
    n = len(x_points)
    result = 0.0
    for i in range(n):
        term = y_points[i]
        for j in range(n):
            if i != j:
                term *= (x_eval - x_points[j]) / (x_points[i] - x_points[j])
        result += term
    return result

print("\nManual Lagrange calculation:")
for i in range(len(x)):
    result = lagrange_manual(x, y, x[i])
    expected = y[i]
    error = abs(result - expected)
    print(f"manual({x[i]}) = {result}, expected {expected}, error = {error}")

# Let's also check what happens with different tolerance levels
print("\nChecking with different tolerance levels:")
import math
for i in range(len(x)):
    result = poly(x[i])
    expected = y[i]

    # Check different tolerance levels
    abs_tol_1e6 = math.isclose(result, expected, rel_tol=1e-6, abs_tol=1e-6)
    abs_tol_1e9 = math.isclose(result, expected, rel_tol=1e-9, abs_tol=1e-9)
    abs_tol_1e12 = math.isclose(result, expected, rel_tol=1e-12, abs_tol=1e-12)

    print(f"x={x[i]}: within 1e-6? {abs_tol_1e6}, within 1e-9? {abs_tol_1e9}, within 1e-12? {abs_tol_1e12}")