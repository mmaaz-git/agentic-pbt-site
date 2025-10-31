#!/usr/bin/env python3
"""Test numpy's argmin/argmax behavior on arrays with all same values"""

import numpy as np

print("Testing numpy's behavior with arrays of all same values:")
print("=" * 60)

test_cases = [
    [0],
    [0, 0, 0],
    [5],
    [5, 5, 5],
    [-1, -1, -1],
    [np.nan],
    [np.nan, np.nan],
]

for arr in test_cases:
    print(f"\nArray: {arr}")
    try:
        print(f"  np.argmin: {np.argmin(arr)}")
        print(f"  np.argmax: {np.argmax(arr)}")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")

# What about the ExtensionArray base class behavior?
print("\n" + "=" * 60)
print("Testing pandas ExtensionArray behavior:")

from pandas.core.arrays.base import ExtensionArray
print(f"ExtensionArray.argmin docstring:")
print(ExtensionArray.argmin.__doc__)