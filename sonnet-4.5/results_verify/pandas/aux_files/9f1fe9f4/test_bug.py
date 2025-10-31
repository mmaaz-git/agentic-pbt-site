#!/usr/bin/env python3
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings

# First, test the hypothesis property test
@given(
    data=st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=1, max_size=100),
    bins=st.integers(min_value=1, max_value=20)
)
@settings(max_examples=500)
def test_cut_bin_count(data, bins):
    arr = np.array(data)
    result = pd.cut(arr, bins)
    assert len(result) == len(arr)

# Test with the specific failing input
print("Testing with reported failing input: data=[0.0, 5e-324], bins=2")
try:
    arr = np.array([0.0, 5e-324])
    result, bins_out = pd.cut(arr, 2, retbins=True)
    print(f"Success: Result = {result}, Bins = {bins_out}")
except ValueError as e:
    print(f"ValueError raised as reported: {e}")

# Let's investigate what's happening with linspace
print("\n--- Investigation of linspace behavior ---")
arr = np.array([0.0, 5e-324])
mn, mx = arr.min(), arr.max()
print(f"min = {mn}, max = {mx}")
print(f"Range = {mx - mn}")

# Simulate what _nbins_to_bins does
nbins = 2
bins = np.linspace(mn, mx, nbins + 1, endpoint=True)
print(f"linspace output: {bins}")
print(f"Unique values: {np.unique(bins)}")

# Apply the adjustment like in the code
adj = (mx - mn) * 0.001  # 0.1% of the range
print(f"\nAdjustment value: {adj}")
print(f"bins[0] = {bins[0]}")
print(f"bins[0] - adj = {bins[0] - adj}")
print(f"Are they equal? {bins[0] == (bins[0] - adj)}")

# Test with duplicates='drop'
print("\n--- Testing with duplicates='drop' ---")
try:
    result2, bins2 = pd.cut(arr, 2, duplicates='drop', retbins=True)
    print(f"Success with duplicates='drop': Result = {result2}, Bins = {bins2}")
except Exception as e:
    print(f"Error even with duplicates='drop': {e}")

# Test with various small ranges
print("\n--- Testing various small ranges ---")
test_cases = [
    (0.0, 1e-324),
    (0.0, 5e-324),
    (0.0, 1e-323),
    (0.0, 1e-320),
    (0.0, 1e-310),
    (0.0, 1e-300),
    (0.0, 1e-15),
    (0.0, 1e-14),
]

for min_val, max_val in test_cases:
    arr = np.array([min_val, max_val])
    try:
        result = pd.cut(arr, 2)
        print(f"Range [{min_val}, {max_val}]: SUCCESS")
    except ValueError:
        print(f"Range [{min_val}, {max_val}]: FAILS with duplicate bins error")

# Run the hypothesis test with a few examples
print("\n--- Running hypothesis test ---")
try:
    test_cut_bin_count()
    print("Hypothesis test passed!")
except Exception as e:
    print(f"Hypothesis test failed: {e}")