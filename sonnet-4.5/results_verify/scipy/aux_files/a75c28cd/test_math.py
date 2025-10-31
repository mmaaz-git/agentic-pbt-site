#!/usr/bin/env python3
"""Verify the mathematical reasoning about bdtr and bdtri"""

import scipy.special as sp
import numpy as np
from scipy.stats import binom

# When k >= n, bdtr sums all probabilities from 0 to n (the entire distribution)
# This should always equal 1.0 regardless of p

print("Mathematical verification:")
print("="*50)
print("When k >= n, bdtr sums the entire binomial distribution")
print("which should always equal 1.0 regardless of p\n")

# Test with different p values when k = n
n = 5
k = 5  # k = n
print(f"Testing with n={n}, k={k} (k = n)")
print("bdtr should be 1.0 for any p value:")

for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
    result = sp.bdtr(k, n, p)
    # Also verify using scipy.stats.binom
    manual_sum = binom.cdf(k, n, p)  # CDF at k
    print(f"  p={p}: bdtr={result}, scipy.stats.binom.cdf={manual_sum}")

print("\n" + "="*50)
print("When k = n, the inverse problem is ill-defined:")
print("If bdtr(k, n, p) = 1.0 for ALL p values,")
print("then bdtri(k, n, 1.0) has infinitely many solutions!\n")

# Show that bdtr approaches 1.0 as k approaches n
print("Testing bdtr as k approaches n:")
n = 10
p = 0.5
for k in [7, 8, 9, 10, 11]:
    result = sp.bdtr(k, n, p)
    print(f"  bdtr({k}, {n}, {p}) = {result}")
    if k >= n:
        print(f"    -> bdtri({k}, {n}, {result}) = {sp.bdtri(k, n, result)}")

print("\n" + "="*50)
print("Edge case: What happens at the boundary y = 1.0?")
n = 10
for k in [8, 9, 10]:
    p_inv = sp.bdtri(k, n, 1.0)
    print(f"  bdtri({k}, {n}, 1.0) = {p_inv}")

# Test y very close to 1.0 but not exactly 1.0
print("\nWhat about y very close to 1.0?")
n = 10
k = 9
for y in [0.999, 0.9999, 0.99999, 0.999999, 1.0]:
    p_inv = sp.bdtri(k, n, y)
    print(f"  bdtri({k}, {n}, {y}) = {p_inv}")