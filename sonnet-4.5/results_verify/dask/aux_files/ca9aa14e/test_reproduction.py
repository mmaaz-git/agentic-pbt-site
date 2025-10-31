#!/usr/bin/env python3
"""Test reproduction of the format_bytes bug"""

from dask.utils import format_bytes

# Test the specific failing case
n = 1_125_899_906_842_624_000
result = format_bytes(n)

print(f"n = {n}")
print(f"n < 2**60 = {n < 2**60}")
print(f"result = {result!r}")
print(f"len(result) = {len(result)}")

# Verify the assertions
try:
    assert n < 2**60
    print("✓ n < 2**60 is True")
except AssertionError:
    print("✗ n < 2**60 is False")

try:
    assert len(result) == 11
    print(f"✓ len(result) == 11")
except AssertionError:
    print(f"✗ len(result) != 11, actual: {len(result)}")

try:
    assert result == '1000.00 PiB'
    print(f"✓ result == '1000.00 PiB'")
except AssertionError:
    print(f"✗ result != '1000.00 PiB', actual: {result}")

# Test a few more edge cases around this boundary
print("\nAdditional tests:")
for multiplier in [999.99, 1000.00, 1000.01]:
    test_n = int(multiplier * 2**50)
    test_result = format_bytes(test_n)
    print(f"  {multiplier:.2f} * 2^50: {test_result!r} (len={len(test_result)})")