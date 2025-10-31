#!/usr/bin/env python3
import numpy as np
import numpy.polynomial as np_poly

# Test case from bug report with tiny leading coefficient
print("Test Case 1: Polynomial with tiny leading coefficient")
print("=" * 60)
coef = [1.0, 1.0, 3.9968426114653685e-66]
p = np_poly.Polynomial(coef)

print(f"Polynomial coefficients: {coef}")
print(f"Polynomial degree: {p.degree()}")
print()

# Compute roots
roots = p.roots()
print(f"Computed roots: {roots}")
print()

# Check if these are actually roots
print("Verification - evaluating p(root) for each root:")
for i, root in enumerate(roots):
    value = p(root)
    print(f"  p(roots[{i}]) = p({root:.6e}) = {value}")
print()

# Check the actual root (should be near -1 for this polynomial)
print("Checking actual root at x = -1.0:")
print(f"  p(-1.0) = {p(-1.0)}")
print()

# Second test case from bug report
print("Test Case 2: Another polynomial with tiny coefficient")
print("=" * 60)
coef2 = [0.0, 1.0, 3.254353641323301e-273]
p2 = np_poly.Polynomial(coef2)

print(f"Polynomial coefficients: {coef2}")
print(f"Polynomial degree: {p2.degree()}")
print()

# Compute roots
roots2 = p2.roots()
print(f"Computed roots: {roots2}")
print()

# Check if these are actually roots
print("Verification - evaluating p(root) for each root:")
for i, root in enumerate(roots2):
    value = p2(root)
    print(f"  p(roots[{i}]) = p({root:.6e}) = {value}")