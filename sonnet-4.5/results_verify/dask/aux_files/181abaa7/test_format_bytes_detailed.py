#!/usr/bin/env python3
"""Detailed analysis of format_bytes behavior"""

from dask.utils import format_bytes

# Test to find exactly where the 10 character limit is exceeded
print("Testing the transition point where output exceeds 10 characters:")
print("=" * 60)

# Test values around 999 * 2**50
test_multipliers = [999, 999.1, 999.2, 999.3, 999.4, 999.5, 999.6, 999.7, 999.8, 999.9, 999.95, 999.99, 1000]

for mult in test_multipliers:
    val = int(mult * 2**50)
    result = format_bytes(val)
    print(f"{mult:7.2f} * 2**50: '{result}' (length: {len(result):2}, {'OK' if len(result) <= 10 else 'VIOLATION'})")

print("\n" + "=" * 60)
print("Testing values approaching 2**60:")
print("=" * 60)

# Test values approaching 2**60
test_values = [
    (2**60 - 2**50, "2**60 - 2**50"),
    (2**60 - 2**40, "2**60 - 2**40"),
    (2**60 - 2**30, "2**60 - 2**30"),
    (2**60 - 1,      "2**60 - 1"),
]

for val, desc in test_values:
    result = format_bytes(val)
    print(f"{desc:15}: '{result}' (length: {len(result):2}, {'OK' if len(result) <= 10 else 'VIOLATION'})")

# Check the actual calculation for the boundary case
print("\n" + "=" * 60)
print("Understanding the calculation:")
print("=" * 60)

n = 1000 * 2**50
print(f"n = 1000 * 2**50 = {n}")
print(f"n / 2**50 = {n / 2**50}")
print(f"Formatted with .2f: {n / 2**50:.2f}")
print(f"Result: '{format_bytes(n)}'")

n = 2**60 - 1
print(f"\nn = 2**60 - 1 = {n}")
print(f"n / 2**50 = {n / 2**50}")
print(f"Formatted with .2f: {n / 2**50:.2f}")
print(f"Result: '{format_bytes(n)}'")

# What's the actual upper bound for 10 characters?
print("\n" + "=" * 60)
print("Finding the actual upper bound for 10 character output:")
print("=" * 60)

# For "XXX.XX PiB" to be exactly 10 characters, we need XXX.XX to be at most 999.99
max_val_for_10_chars = int(999.99 * 2**50)
print(f"Max value for 10 chars: 999.99 * 2**50 = {max_val_for_10_chars}")
print(f"Actual result: '{format_bytes(max_val_for_10_chars)}' (length: {len(format_bytes(max_val_for_10_chars))})")

# One more than that
print(f"Next value: {max_val_for_10_chars + 1}")
result = format_bytes(max_val_for_10_chars + 1)
print(f"Result: '{result}' (length: {len(result)})")

# Compare to 2**60
print(f"\n2**60 = {2**60}")
print(f"Difference: 2**60 - max_val_for_10_chars = {2**60 - max_val_for_10_chars}")