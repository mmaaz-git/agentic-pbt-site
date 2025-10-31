#!/usr/bin/env python3
"""Test the reported Tukey window bug with very small alpha values."""

import numpy as np
from scipy.signal.windows import tukey
from hypothesis import given, strategies as st, settings, example

# First, test the exact reported failing case
print("Testing the exact reported bug case:")
print("=" * 50)
result = tukey(2, alpha=2.225073858507e-311)
print(f"tukey(2, alpha=2.225073858507e-311) = {result}")
print(f"Contains NaN: {np.any(np.isnan(result))}")
print(f"Contains Inf: {np.any(np.isinf(result))}")
print()

# Test with various small alpha values to find the threshold
print("Testing with various small alpha values:")
print("=" * 50)
test_alphas = [1e-100, 1e-200, 1e-300, 1e-305, 1e-308, 1e-310, 1e-311, 1e-312, 2.225073858507e-311]
for alpha in test_alphas:
    result = tukey(2, alpha=alpha)
    has_nan = np.any(np.isnan(result))
    has_inf = np.any(np.isinf(result))
    print(f"alpha={alpha:e}: NaN={has_nan}, Inf={has_inf}, result={result}")

print()
print("Testing division overflow directly:")
print("=" * 50)
# Let's see what happens with the calculation -2.0/alpha
for alpha in [1e-300, 1e-308, 1e-310, 2.225073858507e-311]:
    try:
        value = -2.0/alpha
        cos_value = np.cos(np.pi * value)
        print(f"alpha={alpha:e}: -2.0/alpha={value}, cos(pi*value)={cos_value}")
    except Exception as e:
        print(f"alpha={alpha:e}: Error - {e}")

print()
print("Testing with Hypothesis property test:")
print("=" * 50)

@given(st.integers(min_value=1, max_value=100), st.floats(min_value=0.0, max_value=1.0))
@example(2, 2.225073858507e-311)  # Add the specific failing case
@settings(max_examples=300)
def test_tukey_alpha_range(M, alpha):
    result = tukey(M, alpha=alpha)
    assert len(result) == M, f"Expected length {M}, got {len(result)}"
    assert np.all(np.isfinite(result)), f"Result contains non-finite values for M={M}, alpha={alpha}: {result}"

# Run the hypothesis test
try:
    test_tukey_alpha_range()
    print("Hypothesis test passed all examples")
except AssertionError as e:
    print(f"Hypothesis test failed: {e}")
except Exception as e:
    print(f"Hypothesis test error: {e}")

# Test edge cases
print()
print("Testing edge cases:")
print("=" * 50)
print(f"tukey(2, alpha=0.0) = {tukey(2, alpha=0.0)} (should be rectangular window)")
print(f"tukey(2, alpha=1.0) = {tukey(2, alpha=1.0)} (should be Hann window)")

# Let's also check what a Hann window returns for comparison
from scipy.signal.windows import hann
print(f"hann(2) = {hann(2)} (for comparison)")

# Test if very small alpha behaves like alpha=0
print()
print("Comparing very small alpha with alpha=0:")
print("=" * 50)
rect_window = tukey(10, alpha=0.0)
small_alpha_window = tukey(10, alpha=1e-320)  # Even smaller than the bug threshold
print(f"alpha=0.0: {rect_window}")
print(f"alpha=1e-320: {small_alpha_window}")
print(f"Are they equal? {np.array_equal(rect_window, small_alpha_window)}")