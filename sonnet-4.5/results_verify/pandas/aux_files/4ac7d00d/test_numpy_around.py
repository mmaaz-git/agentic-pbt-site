#!/usr/bin/env python3
"""Test numpy.around behavior with subnormal numbers"""

import numpy as np
import warnings

# Test numpy.around with subnormal numbers
values = [
    1.11253693e-311,  # Very subnormal
    5.56268465e-309,  # Subnormal
    1.11253693e-308,  # Subnormal
    2.225e-308,       # Close to tiny (minimum normal)
    2.3e-308,         # Just above tiny
    1e-10,            # Normal
]

print("Testing np.around with various values:")
for val in values:
    print(f"\nValue: {val}")
    print(f"  Is subnormal: {val < np.finfo(np.float64).tiny}")

    # Try different precisions
    for digits in [3, 10, 100, 313]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = np.around(val, digits)
            print(f"  np.around({val}, {digits}) = {result}")
            if np.isnan(result):
                print(f"    --> Result is NaN!")

print("\n\nThis explains the issue!")
print("np.around() returns NaN for subnormal numbers when digits is too large.")
print("This happens in _round_frac when it computes:")
print("  digits = -int(np.floor(np.log10(abs(frac)))) - 1 + precision")
print("For subnormal values, this results in digits > 308, causing np.around to return NaN.")