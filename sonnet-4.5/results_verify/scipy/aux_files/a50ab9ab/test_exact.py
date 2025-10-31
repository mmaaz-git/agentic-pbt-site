#!/usr/bin/env python3
"""Test the exact failing case from the bug report"""

from scipy.optimize import bisect
import math

def f(x):
    return x**3 - x

# Use the exact values from the bug report
a = -10.0
b = 7.999999999999998

print("Testing the exact case from bug report:")
print(f"a = {a!r}")
print(f"b = {b!r}")
print()

root_ab = bisect(f, a, b)
root_ba = bisect(f, b, a)

print(f"bisect(f, a, b) = {root_ab}")
print(f"bisect(f, b, a) = {root_ba}")
print(f"Difference: {abs(root_ab - root_ba)}")
print()

# Check if they're close
are_close = math.isclose(root_ab, root_ba, rel_tol=1e-9, abs_tol=1e-9)
print(f"Are they close (rel_tol=1e-9, abs_tol=1e-9)? {are_close}")

# Verify both are valid roots
print(f"\nf(root_ab) = f({root_ab}) = {f(root_ab)}")
print(f"f(root_ba) = f({root_ba}) = {f(root_ba)}")

print(f"\nBoth are valid roots? {abs(f(root_ab)) < 1e-9 and abs(f(root_ba)) < 1e-9}")