#!/usr/bin/env python3
"""Analyze what the correct quantile behavior should be"""

import numpy as np
import pandas as pd

print("=" * 60)
print("Analysis of quantile behavior for [0.0]*18 + [1.0, -1.0]")
print("=" * 60)

data = [0.0] * 18 + [1.0, -1.0]
sorted_data = sorted(data)
print(f"Sorted data: {sorted_data}")
print(f"Total elements: {len(data)}")

# Calculate quantiles manually
print("\nManual quantile calculation:")
print("For 2 quantiles, we need percentiles at 0%, 50%, and 100%")

# Using different quantile methods
for method in ['linear', 'lower', 'higher', 'midpoint', 'nearest']:
    q0 = np.quantile(data, 0.0, method=method)
    q50 = np.quantile(data, 0.5, method=method)
    q100 = np.quantile(data, 1.0, method=method)
    print(f"\nMethod '{method}':")
    print(f"  0th percentile: {q0}")
    print(f"  50th percentile: {q50}")
    print(f"  100th percentile: {q100}")

    # Count how many would fall in each bin
    bin1_count = sum(1 for x in data if x <= q50)
    bin2_count = sum(1 for x in data if x > q50)
    print(f"  Bin counts: [{bin1_count}, {bin2_count}]")

print("\n" + "=" * 60)
print("What pandas actually computes:")
print("=" * 60)

s = pd.Series(data)
quantiles = [0.0, 0.5, 1.0]
pandas_quantiles = s.quantile(quantiles)
print(f"Pandas quantiles at {quantiles}: {pandas_quantiles.values}")

print("\n" + "=" * 60)
print("Expected behavior analysis:")
print("=" * 60)

print("For truly equal-sized buckets with 20 items and 2 bins:")
print("- Each bin should have 10 items")
print("- The 50th percentile should be chosen such that 10 items <= split point")
print("- With sorted data: [-1.0] + [0.0]*18 + [1.0]")
print("- The ideal split would be after the 10th element")
print("- The 10th element is 0.0, so the split should include 10 zeros in first bin")
print("- But quantile(0.5) returns 0.0, which puts 19 items in first bin")

print("\nConclusion:")
print("The qcut function is correctly computing quantiles, but this doesn't")
print("produce equal-sized bins when there are many duplicates at the quantile boundary.")