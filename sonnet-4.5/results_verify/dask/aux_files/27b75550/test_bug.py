#!/usr/bin/env python3
"""Test the reported bug in dask resample divisions"""

import pandas as pd
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs

# Test case from bug report
print("=" * 60)
print("Testing the specific failing case from bug report:")
print("=" * 60)

start = pd.Timestamp('2000-01-01')
end = start + pd.Timedelta(days=30)
divisions = pd.date_range(start, end, periods=7)

print(f"Input divisions: {list(divisions)}")
print(f"Number of input divisions: {len(divisions)}")

newdivs, outdivs = _resample_bin_and_out_divs(divisions, 'W', closed='left', label='left')

print(f"\nOutput newdivs: {list(newdivs)}")
print(f"Output outdivs: {list(outdivs)}")

# Check for monotonicity
print("\nChecking monotonicity of outdivs:")
is_monotonic = True
for i in range(len(outdivs) - 1):
    if outdivs[i] >= outdivs[i+1]:
        print(f"ERROR: outdivs[{i}] = {outdivs[i]} >= outdivs[{i+1}] = {outdivs[i+1]}")
        is_monotonic = False

if is_monotonic:
    print("✓ outdivs is strictly monotonic")
else:
    print("✗ outdivs is NOT strictly monotonic - BUG CONFIRMED!")

# Check for duplicates
print("\nChecking for duplicates in outdivs:")
if len(outdivs) != len(set(outdivs)):
    print(f"✗ DUPLICATES FOUND in outdivs!")
    for i in range(len(outdivs) - 1):
        if outdivs[i] == outdivs[i+1]:
            print(f"  Duplicate at positions {i} and {i+1}: {outdivs[i]}")
else:
    print("✓ No duplicates in outdivs")