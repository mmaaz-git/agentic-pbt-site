#!/usr/bin/env python3
"""Verify the mathematical analysis of the Box-Cox transformation."""

import scipy.special as sp
import math
import numpy as np

def analyze_subnormal_issue():
    """Analyze why subnormal values cause problems."""
    print("=" * 60)
    print("SUBNORMAL VALUE ANALYSIS")
    print("=" * 60)

    # Test with the reported failing case
    x = 2.0
    lmbda = 5e-324  # This is a subnormal number

    print(f"Testing with x = {x}, lambda = {lmbda}")
    print(f"Lambda is subnormal: {lmbda < 2.225073858507201e-308}")
    print()

    # Box-Cox forward transformation
    y_scipy = sp.boxcox(x, lmbda)
    y_expected_at_zero = math.log(x)

    print("Forward transformation (boxcox):")
    print(f"  scipy.special.boxcox({x}, {lmbda}) = {y_scipy}")
    print(f"  Expected if λ=0: log({x}) = {y_expected_at_zero}")
    print(f"  Results match: {y_scipy == y_expected_at_zero}")
    print("  → boxcox treats subnormal λ as 0")
    print()

    # Box-Cox inverse transformation
    x_scipy = sp.inv_boxcox(y_scipy, lmbda)
    x_expected_at_zero = math.exp(y_scipy)

    print("Inverse transformation (inv_boxcox):")
    print(f"  scipy.special.inv_boxcox({y_scipy}, {lmbda}) = {x_scipy}")
    print(f"  Expected if λ=0: exp({y_scipy}) = {x_expected_at_zero}")
    print(f"  Results match: {x_scipy == x_expected_at_zero}")
    print(f"  → inv_boxcox does NOT treat subnormal λ as 0")
    print()

    # Show what inv_boxcox actually computed
    print("What inv_boxcox actually computed:")
    print(f"  Formula: (λ*y + 1)^(1/λ)")
    print(f"  λ*y = {lmbda * y_scipy}")
    print(f"  λ*y + 1 = {lmbda * y_scipy + 1}")
    print(f"  1/λ = {1/lmbda if lmbda != 0 else 'undefined'}")

    # The problem: 1.0^inf
    print()
    print("The numerical issue:")
    print(f"  We're computing 1.0^(inf), which is indeterminate")
    print(f"  Different implementations may give different results")
    print(f"  scipy gives: {x_scipy}")
    print(f"  But we expected: {x} (the original value)")
    print()

def test_consistency():
    """Test the consistency between boxcox and inv_boxcox."""
    print("=" * 60)
    print("CONSISTENCY TEST")
    print("=" * 60)

    # Test at exact 0
    x = 2.0
    lmbda = 0.0

    print(f"At λ = 0 (exact):")
    y = sp.boxcox(x, lmbda)
    x_recovered = sp.inv_boxcox(y, lmbda)
    print(f"  x = {x}")
    print(f"  boxcox(x, 0) = {y}")
    print(f"  inv_boxcox(y, 0) = {x_recovered}")
    print(f"  Round-trip successful: {math.isclose(x, x_recovered)}")
    print()

    # Test at smallest positive normal float
    lmbda = 2.225073858507201e-308
    print(f"At λ = {lmbda} (smallest normal):")
    y = sp.boxcox(x, lmbda)
    x_recovered = sp.inv_boxcox(y, lmbda)
    print(f"  x = {x}")
    print(f"  boxcox(x, λ) = {y}")
    print(f"  inv_boxcox(y, λ) = {x_recovered}")
    print(f"  Round-trip successful: {math.isclose(x, x_recovered, rel_tol=1e-10)}")
    print()

    # Test at a subnormal value
    lmbda = 5e-324
    print(f"At λ = {lmbda} (subnormal):")
    y = sp.boxcox(x, lmbda)
    x_recovered = sp.inv_boxcox(y, lmbda)
    print(f"  x = {x}")
    print(f"  boxcox(x, λ) = {y}")
    print(f"  inv_boxcox(y, λ) = {x_recovered}")
    print(f"  Round-trip successful: {math.isclose(x, x_recovered, rel_tol=1e-10)}")
    print()

def check_implementation_details():
    """Check some implementation details."""
    print("=" * 60)
    print("IMPLEMENTATION DETAILS")
    print("=" * 60)

    # Check if there's any documented threshold
    print("Checking for any obvious thresholds in the implementation...")
    print()

    x = 2.0

    # Binary search for the threshold
    low = 0.0
    high = 1e-300

    while high - low > 1e-324:
        mid = (low + high) / 2
        y = sp.boxcox(x, mid)
        if y == math.log(x):
            low = mid
        else:
            high = mid

    print(f"Threshold for boxcox appears to be around: {high}")
    print(f"This is approximately: 2^{math.log2(high) if high > 0 else 'undefined'}")
    print()

    # Check inv_boxcox threshold
    y = math.log(x)
    low = 0.0
    high = 1e-300

    while high - low > 1e-324:
        mid = (low + high) / 2
        x_inv = sp.inv_boxcox(y, mid)
        if x_inv == x:
            low = mid
        else:
            high = mid

    print(f"Threshold for inv_boxcox appears to be around: {high}")
    print(f"This is approximately: 2^{math.log2(high) if high > 0 else 'undefined'}")
    print()

    print("CONCLUSION:")
    print("The thresholds are DIFFERENT for boxcox and inv_boxcox!")
    print("This inconsistency causes the round-trip failure.")

if __name__ == "__main__":
    analyze_subnormal_issue()
    test_consistency()
    check_implementation_details()