#!/usr/bin/env python3
"""Test the reported bisect bug"""

from scipy.optimize import bisect
import math
import numpy as np

def f(x):
    """The test function x^3 - x with roots at -1, 0, 1"""
    return x**3 - x

# Test the specific failing case
a, b = -10.0, 7.999999999999998

print("=== Testing specific failing case ===")
print(f"Interval: [{a}, {b}]")
print(f"f({a}) = {f(a)}")
print(f"f({b}) = {f(b)}")
print(f"Sign check: f(a)*f(b) = {f(a)*f(b)} < 0? {f(a)*f(b) < 0}")
print()

root_ab = bisect(f, a, b)
root_ba = bisect(f, b, a)

print(f"bisect(f, {a}, {b}) = {root_ab}")
print(f"bisect(f, {b}, {a}) = {root_ba}")
print(f"Difference: {abs(root_ab - root_ba)}")
print()

# Verify these are actual roots
print("=== Verifying roots ===")
print(f"f({root_ab}) = {f(root_ab)}")
print(f"f({root_ba}) = {f(root_ba)}")
print()

# Check what roots exist in the interval
print("=== Actual roots in interval ===")
actual_roots = [-1, 0, 1]
for root in actual_roots:
    if min(a, b) <= root <= max(a, b):
        print(f"x = {root}: f({root}) = {f(root)}")
print()

# Test with other root-finding methods
print("=== Testing other methods ===")
from scipy.optimize import brentq, brenth, ridder

try:
    brentq_ab = brentq(f, a, b)
    brentq_ba = brentq(f, b, a)
    print(f"brentq(f, {a}, {b}) = {brentq_ab}")
    print(f"brentq(f, {b}, {a}) = {brentq_ba}")
    print(f"brentq difference: {abs(brentq_ab - brentq_ba)}")
except Exception as e:
    print(f"brentq failed: {e}")

print()

try:
    brenth_ab = brenth(f, a, b)
    brenth_ba = brenth(f, b, a)
    print(f"brenth(f, {a}, {b}) = {brenth_ab}")
    print(f"brenth(f, {b}, {a}) = {brenth_ba}")
    print(f"brenth difference: {abs(brenth_ab - brenth_ba)}")
except Exception as e:
    print(f"brenth failed: {e}")

print()

try:
    ridder_ab = ridder(f, a, b)
    ridder_ba = ridder(f, b, a)
    print(f"ridder(f, {a}, {b}) = {ridder_ab}")
    print(f"ridder(f, {b}, {a}) = {ridder_ba}")
    print(f"ridder difference: {abs(ridder_ab - ridder_ba)}")
except Exception as e:
    print(f"ridder failed: {e}")

# Run the hypothesis test
print("\n=== Running Hypothesis test ===")
from hypothesis import given, strategies as st, assume, settings

@given(
    st.floats(min_value=-10, max_value=-0.1, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.1, max_value=10, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)  # Reduced for faster testing
def test_interval_direction_invariance(a, b):
    assume(abs(a - b) > 0.5)

    def f(x):
        return x**3 - x

    fa, fb = f(a), f(b)
    assume(fa * fb < 0)

    root_ab = bisect(f, a, b)
    root_ba = bisect(f, b, a)

    assert math.isclose(root_ab, root_ba, rel_tol=1e-9, abs_tol=1e-9), \
        f"bisect gave different results: f({a},{b})={root_ab} vs f({b},{a})={root_ba}"

try:
    test_interval_direction_invariance()
    print("Hypothesis test passed (no failures found)")
except AssertionError as e:
    print(f"Hypothesis test failed: {e}")