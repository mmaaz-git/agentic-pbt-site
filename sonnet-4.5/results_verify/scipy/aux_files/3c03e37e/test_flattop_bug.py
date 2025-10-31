#!/usr/bin/env python3
"""Test script to reproduce the flattop window bug"""

import numpy as np
import scipy.signal.windows as windows
from hypothesis import given, strategies as st, settings

# First, run the exact reproduction code from the bug report
print("=" * 60)
print("REPRODUCING BUG REPORT")
print("=" * 60)

w = windows.flattop(3)
print(f"Window values for M=3: {w}")
print(f"Max value: {np.max(w):.20f}")
print(f"Expected: <= 1.0")
print(f"Exceeds 1.0? {np.max(w) > 1.0}")

a = np.array([0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368])
print(f"\nFlattop coefficients sum: {np.sum(a):.20f}")

# Now test with several odd and even values
print("\n" + "=" * 60)
print("TESTING VARIOUS VALUES OF M")
print("=" * 60)

for M in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 25, 50, 51, 100, 101]:
    w = windows.flattop(M)
    max_val = np.max(w)
    print(f"M={M:3d}: max={max_val:.20f}, exceeds 1.0? {max_val > 1.0}")

# Now run the hypothesis test
print("\n" + "=" * 60)
print("RUNNING HYPOTHESIS TEST")
print("=" * 60)

failing_cases = []

@given(st.integers(min_value=1, max_value=1000))
@settings(max_examples=200)
def test_max_value_normalized_to_one(M):
    w = windows.flattop(M)
    max_val = np.max(w)
    if not (np.isclose(max_val, 1.0, rtol=1e-10, atol=1e-14) or max_val < 1.0):
        failing_cases.append((M, max_val))
        # Don't raise the assertion to collect all failures
        # raise AssertionError(f"flattop({M}) has max value {max_val}, expected <= 1.0")

# Run the test
try:
    test_max_value_normalized_to_one()
    print(f"Hypothesis test completed. Found {len(failing_cases)} failing cases.")
except Exception as e:
    print(f"Hypothesis test raised exception: {e}")

if failing_cases:
    print(f"\nFirst 10 failing cases:")
    for M, max_val in failing_cases[:10]:
        print(f"  M={M}: max_val={max_val:.20f}")

    # Check pattern
    odd_failures = [M for M, _ in failing_cases if M % 2 == 1]
    even_failures = [M for M, _ in failing_cases if M % 2 == 0]
    print(f"\nPattern analysis:")
    print(f"  Odd M failures: {len(odd_failures)}")
    print(f"  Even M failures: {len(even_failures)}")

# Let's verify the mathematical analysis from the bug report
print("\n" + "=" * 60)
print("VERIFYING MATHEMATICAL ANALYSIS")
print("=" * 60)

# For odd M with sym=True, the center point should be at index (M-1)//2
# At this point, all cosine terms should equal 1
M = 3
w = windows.flattop(M)
center_idx = (M - 1) // 2
print(f"For M=3, center index is {center_idx}")
print(f"Window value at center: {w[center_idx]:.20f}")

# Manually compute what the center value should be
# At the center of an odd-length symmetric window, fac = 0
# cos(0) = 1, cos(k*0) = 1 for all k
# So the window value equals sum of coefficients
manual_sum = sum([0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368])
print(f"Manual sum of coefficients: {manual_sum:.20f}")
print(f"Matches window center value? {np.isclose(w[center_idx], manual_sum)}")

# Test with corrected coefficients
print("\n" + "=" * 60)
print("TESTING PROPOSED FIX")
print("=" * 60)

original = [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368]
corrected = [x / sum(original) for x in original]
print(f"Original coefficients sum: {sum(original):.20f}")
print(f"Corrected coefficients sum: {sum(corrected):.20f}")
print(f"Corrected coefficients: {corrected}")

# Simulate the corrected implementation
def flattop_corrected(M):
    """Simulated corrected flattop function"""
    if M < 1:
        return np.array([])
    if M == 1:
        return np.ones(1)

    corrected_a = np.array([0.215578947368421, 0.416631578947368, 0.277263157894737,
                           0.083578947368421, 0.006947368421053])

    # Simplified version of _general_cosine_impl for sym=True
    fac = np.linspace(-np.pi, np.pi, M)
    w = np.zeros(M)
    for k in range(len(corrected_a)):
        w += corrected_a[k] * np.cos(k * fac)
    return w

# Test corrected version
for M in [3, 5, 7, 9, 11]:
    w_corrected = flattop_corrected(M)
    max_val = np.max(w_corrected)
    print(f"Corrected M={M}: max={max_val:.20f}, exceeds 1.0? {max_val > 1.0}")