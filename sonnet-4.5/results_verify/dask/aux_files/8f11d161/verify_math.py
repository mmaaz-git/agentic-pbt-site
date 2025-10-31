#!/usr/bin/env python3
"""Verify the mathematical claims in the bug report"""

from dask.utils import format_bytes

# Check 2**60 value
print("2**60 value and related calculations:")
print(f"2**60 = {2**60}")
print(f"2**50 (1 PiB) = {2**50}")
print(f"1000 * 2**50 (1000 PiB) = {1000 * 2**50}")
print(f"1024 * 2**50 (1024 PiB) = {1024 * 2**50}")
print()

# Check if 1000 PiB < 2**60
print("Checking boundary conditions:")
print(f"1000 PiB < 2**60: {1000 * 2**50 < 2**60}")
print(f"1024 PiB < 2**60: {1024 * 2**50 < 2**60}")
print()

# Test the exact boundary where 11 chars start appearing
print("Finding the exact boundary where output exceeds 10 characters:")

def find_boundary():
    """Binary search to find the exact value where output length exceeds 10 chars"""
    low = 0
    high = 2**60 - 1

    while low < high:
        mid = (low + high) // 2
        if len(format_bytes(mid)) <= 10:
            low = mid + 1
        else:
            high = mid

    return low

boundary = find_boundary()
print(f"First value with >10 chars: {boundary}")
print(f"  Format: '{format_bytes(boundary)}' (length: {len(format_bytes(boundary))})")
print(f"  As PiB: {boundary / 2**50:.2f} PiB")
print(f"  < 2**60: {boundary < 2**60}")
print()

# Check the value just before
before = boundary - 1
print(f"Last value with <=10 chars: {before}")
print(f"  Format: '{format_bytes(before)}' (length: {len(format_bytes(before))})")
print(f"  As PiB: {before / 2**50:.2f} PiB")
print()

# Analyze the formatting pattern
print("Analyzing format patterns for values >= 1000 in their unit:")
test_cases = [
    (999.99, "PiB"),
    (1000.00, "PiB"),
    (1000.01, "PiB"),
    (1023.99, "PiB"),
    (1024.00, "PiB"),
]

for value, unit in test_cases:
    formatted = f"{value:.2f} {unit}"
    print(f"  {value:7.2f} {unit} -> '{formatted}' (length: {len(formatted)})")
print()

# Check how the current implementation handles edge cases
print("Checking the 0.9 threshold behavior:")
for prefix, k in [("Pi", 2**50), ("Ti", 2**40), ("Gi", 2**30), ("Mi", 2**20), ("ki", 2**10)]:
    threshold = k * 0.9
    just_below = int(threshold - 1)
    at_threshold = int(threshold)

    print(f"  {prefix}B threshold (k * 0.9): {threshold:.0f}")
    print(f"    Just below: {format_bytes(just_below)}")
    print(f"    At threshold: {format_bytes(at_threshold)}")