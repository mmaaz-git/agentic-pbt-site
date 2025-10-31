#!/usr/bin/env python3
"""Test numerical stability expectations for polynomial division."""

import numpy as np
from numpy.polynomial import Polynomial

def test_different_magnitudes():
    """Test with coefficients of very different magnitudes."""
    print("=== Testing different magnitude combinations ===")

    test_cases = [
        # (a_coef, b_coef, description)
        ([0]*7 + [1], [72, 1.75], "Original bug case"),
        ([0]*7 + [1], [1, 1], "Unit coefficients"),
        ([0]*7 + [1], [100, 1], "100:1 ratio"),
        ([0]*7 + [1], [1000, 1], "1000:1 ratio"),
        ([0]*7 + [1], [10000, 1], "10000:1 ratio"),
        ([0]*6 + [1], [72, 1.75], "Degree 6 with same divisor"),
        ([0]*5 + [1], [72, 1.75], "Degree 5 with same divisor"),
        ([0]*4 + [1], [72, 1.75], "Degree 4 with same divisor"),
        ([0]*3 + [1], [72, 1.75], "Degree 3 with same divisor"),
    ]

    for a_coef, b_coef, desc in test_cases:
        a = Polynomial(a_coef)
        b = Polynomial(b_coef)
        q, r = divmod(a, b)
        reconstructed = b * q + r

        # Calculate error
        if len(reconstructed.coef) == len(a.coef):
            error = np.max(np.abs(reconstructed.coef - a.coef))
        else:
            max_len = max(len(reconstructed.coef), len(a.coef))
            rec_pad = np.pad(reconstructed.coef, (0, max_len - len(reconstructed.coef)))
            a_pad = np.pad(a.coef, (0, max_len - len(a.coef)))
            error = np.max(np.abs(rec_pad - a_pad))

        print(f"{desc:30} Error: {error:.2e}")

test_different_magnitudes()

print("\n=== Checking condition number ===")
# The condition number gives us an idea of numerical stability
# For the case that fails
a = Polynomial([0]*7 + [1])
b = Polynomial([72, 1.75])

# Get the quotient
q, r = divmod(a, b)

print(f"Quotient max coefficient: {np.max(np.abs(q.coef)):.2e}")
print(f"Remainder value: {np.abs(r.coef[0]):.2e}")
print(f"Ratio of remainder to input: {np.abs(r.coef[0]) / 1:.2e}")

# Compare with a well-conditioned case
a2 = Polynomial([0]*7 + [1])
b2 = Polynomial([1, 1])
q2, r2 = divmod(a2, b2)

print(f"\nWell-conditioned case:")
print(f"Quotient max coefficient: {np.max(np.abs(q2.coef)):.2e}")
print(f"Remainder value: {np.abs(r2.coef[0]):.2e} if non-zero, else 0")

print("\n=== Testing error accumulation pattern ===")
# The error pattern suggests floating point accumulation
# Let's check if the errors follow a pattern

a = Polynomial([0]*7 + [1])
b = Polynomial([72, 1.75])
q, r = divmod(a, b)
reconstructed = b * q + r

errors = reconstructed.trim().coef - a.coef

print("Error coefficients:")
for i, e in enumerate(errors):
    if e != 0:
        print(f"  x^{i}: {e:.6e}")

# Check if errors decrease with power (typical of accumulation)
non_zero_errors = [abs(e) for e in errors if e != 0]
if len(non_zero_errors) > 1:
    ratios = [non_zero_errors[i]/non_zero_errors[i+1] for i in range(len(non_zero_errors)-1)]
    print(f"\nError ratios (each/next): {ratios}")
    print(f"Average ratio: {np.mean(ratios):.2f}")

    if all(r > 10 for r in ratios):
        print("Errors decrease exponentially - typical of floating point accumulation")