#!/usr/bin/env python3
"""Test how pandas handles the same resample case"""

import pandas as pd
import numpy as np

# Create the same test data
dates = pd.date_range('2000-12-17 00:00:00', periods=2, freq='30min')
df = pd.DataFrame({'value': [1, 2]}, index=dates)

print("Original DataFrame:")
print(df)
print(f"Index: {df.index.tolist()}")

# Apply resample with same parameters
print("\n" + "="*60)
print("Resampling with '1W', closed='right', label='right'...")
resampler = df.resample('1W', closed='right', label='right')

# Get the bins/groups
print("\nResampler groups:")
for name, group in resampler:
    print(f"  Bin {name}: {group.index.tolist()}")

# Sum the results
result = resampler.sum()
print(f"\nResult of resample().sum():")
print(result)
print(f"Result index: {result.index.tolist()}")

# Check if the resulting index is monotonic
print(f"\nIs result index monotonic? {result.index.is_monotonic_increasing}")

# Now let's understand what the bins should be
print("\n" + "="*60)
print("Understanding the bins:")
print(f"  Original data range: {dates[0]} to {dates[-1]}")
print(f"  Weekly frequency with closed='right' means bins like (start, end]")
print(f"  label='right' means the bin is labeled with its right edge")