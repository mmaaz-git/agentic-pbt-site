#!/usr/bin/env python3
"""Test script to reproduce the reported qcut bugs"""

import pandas as pd
from pandas.core.reshape.tile import qcut
import numpy as np

print("=" * 60)
print("Bug 1: Crash with tiny quantile differences")
print("=" * 60)
data_crash = [0.0] * 19 + [2.2250738585e-313]
print(f"Data: [0.0]*19 + [2.2250738585e-313]")
print(f"Length: {len(data_crash)}, Unique values: {len(set(data_crash))}")

try:
    result, bins = qcut(data_crash, 2, retbins=True, duplicates='drop')
    print("SUCCESS: No crash occurred")
    print(f"Bins: {bins}")
    print(f"Bin counts: {result.value_counts()}")
except ValueError as e:
    print(f"ERROR (ValueError): {e}")
except Exception as e:
    print(f"ERROR (Other): {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("Bug 2: Severely unequal bin sizes")
print("=" * 60)
data_unequal = [0.0] * 18 + [1.0, -1.0]
print(f"Data: [0.0]*18 + [1.0, -1.0]")
print(f"Length: {len(data_unequal)}, Unique values: {len(set(data_unequal))}")

try:
    result, bins = qcut(data_unequal, 2, retbins=True, duplicates='drop')
    print("SUCCESS: No crash")
    print(f"Bins returned: {bins}")
    print(f"Bin counts:\n{result.value_counts()}")

    # Calculate expected vs actual
    total_non_nan = result.notna().sum()
    num_bins = len(result.value_counts())
    expected_per_bin = total_non_nan / num_bins if num_bins > 0 else 0
    print(f"\nTotal non-NaN values: {total_non_nan}")
    print(f"Number of bins: {num_bins}")
    print(f"Expected per bin: ~{expected_per_bin:.1f}")

    # Check the ratio
    counts = result.value_counts().values
    if len(counts) >= 2:
        max_ratio = max(counts) / min(counts)
        print(f"Ratio of largest to smallest bin: {max_ratio:.1f}:1")

except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("Running property-based test from the bug report")
print("=" * 60)

from hypothesis import given, strategies as st, assume

@given(st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=20, max_size=200),
       st.integers(min_value=2, max_value=10))
def test_qcut_equal_sized_bins(data, num_quantiles):
    assume(len(set(data)) >= num_quantiles)

    result, bins = qcut(data, num_quantiles, retbins=True, duplicates='drop')

    unique_bins = result[pd.notna(result)].unique()
    assume(len(unique_bins) >= 2)

    bin_counts = result.value_counts()

    expected_count = len([x for x in data if not pd.isna(x)]) / len(unique_bins)
    tolerance = expected_count * 0.5

    for count in bin_counts:
        assert abs(count - expected_count) <= tolerance, f"Bin count {count} too far from expected {expected_count}"

print("Running property-based tests with specific failing inputs...")

# Test with the reported failing inputs
try:
    print("\nTest 1: data=[0.0]*19 + [2.2250738585e-313], num_quantiles=2")
    test_qcut_equal_sized_bins([0.0]*19 + [2.2250738585e-313], 2)
    print("  PASSED")
except AssertionError as e:
    print(f"  FAILED (assertion): {e}")
except Exception as e:
    print(f"  FAILED (exception): {type(e).__name__}: {e}")

try:
    print("\nTest 2: data=[0.0]*18 + [1.0, -1.0], num_quantiles=2")
    test_qcut_equal_sized_bins([0.0]*18 + [1.0, -1.0], 2)
    print("  PASSED")
except AssertionError as e:
    print(f"  FAILED (assertion): {e}")
except Exception as e:
    print(f"  FAILED (exception): {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("Additional edge case testing")
print("=" * 60)

# Test what happens with all identical values
print("\nTest 3: All identical values")
data_identical = [1.0] * 20
try:
    result, bins = qcut(data_identical, 2, retbins=True, duplicates='drop')
    print(f"  Result: {result.value_counts()}")
    print(f"  Bins: {bins}")
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")

# Test with only 2 unique values
print("\nTest 4: Only 2 unique values, requesting 2 quantiles")
data_two_unique = [0.0] * 10 + [1.0] * 10
try:
    result, bins = qcut(data_two_unique, 2, retbins=True, duplicates='drop')
    print(f"  Bin counts: {result.value_counts()}")
    print(f"  Bins: {bins}")
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")