#!/usr/bin/env python3
"""Test to reproduce the format_bytes bug."""

from dask.utils import format_bytes

# Test the specific failing input
n = 1125894277343089729
result = format_bytes(n)
print(f"format_bytes({n}) = '{result}'")
print(f"Length: {len(result)}")
print(f"n < 2**60: {n < 2**60}")
print()

# Test additional values around this boundary
test_values = [
    1000 * 2**50,  # Exactly 1000 PiB
    999 * 2**50,   # 999 PiB
    1001 * 2**50,  # 1001 PiB
    1100 * 2**50,  # 1100 PiB
]

print("Additional test values:")
for val in test_values:
    result = format_bytes(val)
    print(f"format_bytes({val}) = '{result}' (length: {len(result)})")