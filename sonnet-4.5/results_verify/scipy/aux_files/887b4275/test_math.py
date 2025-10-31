#!/usr/bin/env python3
"""Test the mathematical validity of polynomial division with leading zeros"""

import numpy as np

print("Testing mathematical validity of polynomial division with leading zeros")
print("=" * 70)

# In polynomial representation, [0, 1] represents 0*x + 1 = 1 (constant polynomial)
# And [1, 2] represents 1*x + 2

print("\nPolynomial interpretation:")
print("divisor [0, 1] represents: 0*x + 1 = 1 (a constant)")
print("signal [1, 2] represents: 1*x + 2")

print("\nDividing (1*x + 2) by 1:")
print("Expected quotient: 1*x + 2")
print("Expected remainder: 0")

print("\nManual polynomial division:")
# If we interpret this as polynomial division:
# (1*x + 2) / 1 = 1*x + 2 with remainder 0

print("\n" + "=" * 70)
print("Alternative interpretation - leading zero means lower degree:")
print("\nIf [0, 1] is meant to represent just '1' (degree 0 polynomial)")
print("Then dividing [1, 2] by [1] should give:")

# Using numpy polynomial operations with proper degree
print("\nUsing numpy.polynomial.polynomial.polydiv:")
from numpy.polynomial import polynomial as P

# Coefficients are in ascending order for polynomial
p1 = [2, 1]  # 2 + 1*x
p2 = [1]     # just 1

q, r = P.polydiv(p1, p2)
print(f"Dividing {p1} by {p2}")
print(f"Quotient: {q}")
print(f"Remainder: {r}")

print("\n" + "=" * 70)
print("The issue: scipy.signal.deconvolve uses lfilter internally")
print("lfilter expects a[0] != 0 for the filter coefficient")
print("This is a signal processing constraint, not a polynomial math constraint")

print("\n" + "=" * 70)
print("Testing if leading zeros can be stripped:")

def safe_deconvolve(signal, divisor):
    """Deconvolve after stripping leading zeros from divisor"""
    # Find first non-zero element
    first_nonzero = np.argmax(np.abs(divisor) > 1e-10)
    if first_nonzero > 0:
        divisor = divisor[first_nonzero:]

    # Now try regular polynomial division
    return np.polydiv(signal, divisor)

signal = [1.0, 2.0]
divisor = [0.0, 1.0]

print(f"\nOriginal divisor: {divisor}")
print(f"After stripping leading zeros: {divisor[1:]}")

q, r = safe_deconvolve(signal, divisor)
print(f"\nQuotient: {q}")
print(f"Remainder: {r}")

# Verify the result
import scipy.signal
reconstructed = scipy.signal.convolve([1.0], q) + r
print(f"\nReconstructed signal: {reconstructed}")
print(f"Original signal: {signal}")
print(f"Match: {np.allclose(reconstructed, signal)}")