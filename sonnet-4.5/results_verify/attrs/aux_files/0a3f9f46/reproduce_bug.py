#!/usr/bin/env python3
"""Reproduce the bug reported in dask.dataframe.tseries.resample._resample_bin_and_out_divs"""

import pandas as pd
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs

print("=" * 80)
print("Testing the specific failing case from the bug report")
print("=" * 80)

# Test case from the bug report
divisions = [pd.Timestamp('2000-12-17 00:00:00'), pd.Timestamp('2000-12-17 01:00:00')]
print(f"Input divisions: {divisions}")
print(f"Rule: '1W', closed='right', label='right'")

newdivs, outdivs = _resample_bin_and_out_divs(divisions, '1W', closed='right', label='right')

print(f"\nOutput:")
print(f"newdivs: {newdivs}")
print(f"outdivs: {outdivs}")
print(f"outdivs[0] = {outdivs[0]}")
print(f"outdivs[1] = {outdivs[1]}")
print(f"Is monotonic? {outdivs[0] <= outdivs[1]}")

if outdivs[0] > outdivs[1]:
    print("\n❌ BUG CONFIRMED: outdivs is NOT monotonic!")
    print(f"   outdivs[0] ({outdivs[0]}) > outdivs[1] ({outdivs[1]})")
else:
    print("\n✓ No bug: outdivs is monotonic")