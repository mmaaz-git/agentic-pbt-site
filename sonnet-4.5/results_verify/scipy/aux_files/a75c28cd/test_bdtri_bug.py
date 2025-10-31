#!/usr/bin/env python3
"""Test the reported bdtri bug"""

import numpy as np
import scipy.special as sp

# Test case from bug report: k=1, n=1, p=0.5
k, n, p = 1, 1, 0.5

print("Testing the reported bug case where k >= n:")
print(f"Input: k={k}, n={n}, p={p}")
print()

# Step 1: Calculate bdtr
y = sp.bdtr(k, n, p)
print(f"bdtr({k}, {n}, {p}) = {y}")

# Step 2: Try to reconstruct p using bdtri
p_reconstructed = sp.bdtri(k, n, y)
print(f"bdtri({k}, {n}, {y}) = {p_reconstructed}")

# Step 3: Calculate bdtr with reconstructed p
y_reconstructed = sp.bdtr(k, n, p_reconstructed)
print(f"bdtr({k}, {n}, {p_reconstructed}) = {y_reconstructed}")

# Check if round trip works
print()
try:
    assert np.isclose(y, y_reconstructed, rtol=1e-8, atol=1e-10)
    print("Round-trip test PASSED")
except AssertionError:
    print("Round-trip test FAILED: values are not close")
    print(f"  Original y: {y}")
    print(f"  Reconstructed y: {y_reconstructed}")

# Let's test more cases where k >= n
print("\n" + "="*50)
print("Testing additional cases where k >= n:\n")

test_cases = [
    (2, 2, 0.3),  # k = n
    (3, 2, 0.5),  # k > n
    (5, 3, 0.7),  # k > n
    (0, 0, 0.5),  # k = n = 0
]

for k, n, p in test_cases:
    print(f"Test case: k={k}, n={n}, p={p}")
    y = sp.bdtr(k, n, p)
    p_reconstructed = sp.bdtri(k, n, y)
    print(f"  bdtr({k}, {n}, {p}) = {y}")
    print(f"  bdtri({k}, {n}, {y}) = {p_reconstructed}")
    print()

# Test case where k < n (should work)
print("="*50)
print("Testing a case where k < n (should work):\n")
k, n, p = 3, 10, 0.5
print(f"Test case: k={k}, n={n}, p={p}")
y = sp.bdtr(k, n, p)
p_reconstructed = sp.bdtri(k, n, y)
y_reconstructed = sp.bdtr(k, n, p_reconstructed)
print(f"  bdtr({k}, {n}, {p}) = {y}")
print(f"  bdtri({k}, {n}, {y}) = {p_reconstructed}")
print(f"  bdtr({k}, {n}, {p_reconstructed}) = {y_reconstructed}")
try:
    assert np.isclose(y, y_reconstructed, rtol=1e-8, atol=1e-10)
    print("  Round-trip test PASSED")
except AssertionError:
    print("  Round-trip test FAILED")