#!/usr/bin/env python3
"""Debug the label formatting issue"""

import numpy as np
import pandas as pd
from pandas import Index
import warnings

# Test _round_frac with the problematic value
value = 1.11253693e-311
print(f"Testing np.log10 with tiny value: {value}")
print(f"  abs(value): {abs(value)}")
try:
    log_val = np.log10(abs(value))
    print(f"  np.log10(abs(value)): {log_val}")
    floor_val = np.floor(log_val)
    print(f"  np.floor(log_val): {floor_val}")
    digits = -int(floor_val) - 1 + 3
    print(f"  digits for precision=3: {digits}")
except RuntimeWarning as e:
    print(f"  RuntimeWarning: {e}")
except Exception as e:
    print(f"  ERROR: {e}")

# Check if the issue is in _infer_precision
print("\n\nChecking _infer_precision logic:")
bins = np.array([-1.11253693e-311, 5.56268465e-309, 1.11253693e-308])
base_precision = 3

for precision in range(base_precision, 10):
    print(f"\nPrecision {precision}:")
    rounded_bins = []
    for b in bins:
        if not np.isfinite(b) or b == 0:
            rounded = b
        else:
            frac, whole = np.modf(b)
            if whole == 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    log_result = np.log10(abs(frac))
                    if np.isfinite(log_result):
                        digits = -int(np.floor(log_result)) - 1 + precision
                    else:
                        # log10 failed due to underflow
                        digits = 308 + precision  # Use a large value for subnormal numbers
            else:
                digits = precision
            rounded = np.around(b, digits)
        rounded_bins.append(rounded)

    rounded_bins = np.array(rounded_bins)
    print(f"  Rounded bins: {rounded_bins}")
    diffs = np.diff(rounded_bins)
    print(f"  Diffs: {diffs}")
    print(f"  All diffs != 0? {np.all(diffs != 0)}")
    if np.all(diffs != 0):
        print(f"  --> Would use precision {precision}")
        break

# The real issue - check why cut returns NaN
print("\n\nThe real issue:")
print("When bins are too close together relative to float precision,")
print("the IntervalIndex creation or category mapping might fail.")
print("\nThe warning 'invalid value encountered in divide' suggests")
print("there's a division by a very small number somewhere.")

# Check where division happens in pandas
print("\nLet's check where pandas might divide by the range:")
data_range = 1.11253693e-308
adj = 0.001 * data_range
print(f"Range: {data_range}")
print(f"0.1% adjustment: {adj}")
print(f"adj is subnormal: {adj < 2.2250738585072014e-308}")

# Could be in interval width calculations
interval_width = data_range / 2  # for 2 bins
print(f"\nInterval width for 2 bins: {interval_width}")
print(f"Is subnormal: {interval_width < 2.2250738585072014e-308}")