#!/usr/bin/env python3
"""Test the reported bug in dask.utils.format_bytes"""

from dask.utils import format_bytes

# Test the specific failing input
n = 1125894277343089729
result = format_bytes(n)

print(f"Testing n = {n}")
print(f"n < 2**60? {n < 2**60} (2**60 = {2**60})")
print(f"format_bytes({n}) = '{result}'")
print(f"Length: {len(result)}")
print(f"Length <= 10? {len(result) <= 10}")
print()

# Let's test a few more values near 2**60
print("Testing values near 2**60:")
test_values = [
    2**60 - 1,  # Maximum value that should satisfy the property
    2**60 - 1000,
    2**50 * 1000,  # Around 1000 PiB
    2**50 * 1024,  # Around 1024 PiB
    2**50 * 900,   # Around 900 PiB
]

for val in test_values:
    res = format_bytes(val)
    print(f"  format_bytes({val:20}) = '{res:12}' (len={len(res):2}, valid={val < 2**60})")

# Let's understand the thresholds better
print("\nUnderstanding thresholds:")
print(f"PiB threshold (2**50 * 0.9): {2**50 * 0.9:.0f}")
print(f"Value that gives 1000.00 PiB: {2**50 * 1000:.0f}")
print(f"Value that gives 1024.00 PiB: {2**50 * 1024:.0f}")
print(f"2**60: {2**60}")