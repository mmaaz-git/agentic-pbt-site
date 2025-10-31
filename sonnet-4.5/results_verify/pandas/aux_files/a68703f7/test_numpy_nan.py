#!/usr/bin/env python3
"""Test NumPy's documented behavior with NaN values in mean calculations"""

import numpy as np

print("=" * 60)
print("NUMPY MEAN BEHAVIOR WITH NaN VALUES")
print("=" * 60)

# Test various cases with NaN
test_cases = [
    np.array([1.0, 2.0, 3.0]),           # No NaN
    np.array([1.0, np.nan]),              # One NaN
    np.array([np.nan, np.nan]),           # All NaN
    np.array([1.0, 2.0, np.nan, 3.0]),    # NaN in middle
    np.array([np.nan]),                   # Single NaN
]

print("\nStandard numpy.mean() behavior:")
print("-" * 40)
for arr in test_cases:
    mean_val = np.mean(arr)
    print(f"Array: {arr}")
    print(f"Mean: {mean_val}\n")

print("\nnumpy.nanmean() behavior (ignores NaN):")
print("-" * 40)
for arr in test_cases:
    nanmean_val = np.nanmean(arr)
    print(f"Array: {arr}")
    print(f"NanMean: {nanmean_val}\n")

# Test pandas Series behavior for comparison
print("\n" + "=" * 60)
print("PANDAS SERIES BEHAVIOR WITH NaN VALUES")
print("=" * 60)

import pandas as pd

for data in [[1.0, np.nan], [1.0, 2.0, np.nan, 3.0], [np.nan, np.nan]]:
    series = pd.Series(data)
    print(f"Series: {list(data)}")
    print(f"Mean (skipna=True, default): {series.mean()}")
    print(f"Mean (skipna=False): {series.mean(skipna=False)}\n")

# Check if this is standard IEEE 754 floating point behavior
print("=" * 60)
print("IEEE 754 FLOATING POINT STANDARD BEHAVIOR")
print("=" * 60)

print("According to IEEE 754 standard:")
print("- Any arithmetic operation with NaN returns NaN")
print("- This includes mean calculations")
print("\nSimple test: (1.0 + NaN) / 2 =", (1.0 + np.nan) / 2)