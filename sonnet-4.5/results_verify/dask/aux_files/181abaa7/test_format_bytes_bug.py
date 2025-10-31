#!/usr/bin/env python3
"""Test to reproduce the format_bytes bug"""

from dask.utils import format_bytes

# Test the specific failing case
n = 2**60 - 1
result = format_bytes(n)
print(f"format_bytes({n}) = '{result}'")
print(f"Length: {len(result)} characters")
print(f"Expected: <= 10 characters (as documented)")

try:
    assert len(result) <= 10, f"Constraint violated: {len(result)} > 10"
    print("Assertion passed")
except AssertionError as e:
    print(f"AssertionError: {e}")

# Test some edge cases around the threshold
test_values = [
    1000 * 2**50 - 1,  # Just below the threshold where issue starts
    1000 * 2**50,      # At the threshold
    2**60 - 1,          # Maximum value claimed to work
]

print("\nTesting edge cases:")
for val in test_values:
    result = format_bytes(val)
    print(f"format_bytes({val:20}) = '{result:15}' (length: {len(result)})")