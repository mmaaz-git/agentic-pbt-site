#!/usr/bin/env python3
"""Systematically find all violations of the 10-character guarantee"""

from dask.utils import format_bytes

# Find the threshold where violations start for each unit
units = [
    ("PiB", 2**50),
    ("TiB", 2**40),
    ("GiB", 2**30),
    ("MiB", 2**20),
    ("kiB", 2**10),
]

print("Finding violation thresholds for each unit:")
print("-" * 70)

violations_found = []

for unit_name, unit_size in units:
    # The problem occurs when n >= k * 0.9 but formats to >= 1000.00
    # This happens when n/k >= 999.995 (rounds to 1000.00)
    threshold = int(unit_size * 999.995)

    # Test around this threshold
    for offset in range(-10, 10):
        test_val = threshold + offset
        if test_val < 0 or test_val >= 2**60:
            continue

        result = format_bytes(test_val)
        if len(result) > 10:
            violations_found.append((test_val, result))
            print(f"{unit_name}: format_bytes({test_val}) = {result!r} (len={len(result)})")
            break

print(f"\nTotal violations found: {len(violations_found)}")

# Check the specific reported case
print("\n" + "-" * 70)
print("Verifying the reported case:")
n = 1125894277343089729
result = format_bytes(n)
print(f"format_bytes({n}) = {result!r}")
print(f"n / 2**50 = {n / 2**50:.6f}")
print(f"This rounds to: {n / 2**50:.2f}")

# Find the exact boundary
print("\n" + "-" * 70)
print("Finding exact violation boundaries:")
for unit_name, unit_size in units:
    # Binary search for the exact boundary
    low = int(unit_size * 999.994)
    high = int(unit_size * 999.996)

    if high >= 2**60:
        continue

    while low < high:
        mid = (low + high) // 2
        if len(format_bytes(mid)) > 10:
            high = mid
        else:
            low = mid + 1

    if low < 2**60 and len(format_bytes(low)) > 10:
        result = format_bytes(low)
        print(f"{unit_name}: First violation at {low} => {result!r}")
        print(f"      {low} / {unit_size} = {low / unit_size:.10f}")