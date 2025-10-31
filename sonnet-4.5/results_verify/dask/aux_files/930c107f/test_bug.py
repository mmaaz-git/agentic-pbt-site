#!/usr/bin/env python3
"""Test the format_bytes bug report"""

from dask.utils import format_bytes

# Test case from the bug report
n = 1125899906842624000
result = format_bytes(n)

print(f"format_bytes({n}) = '{result}'")
print(f"Length: {len(result)}")
print(f"n < 2**60: {n < 2**60}")

# Let's also test the boundary more carefully
print("\n--- Testing boundaries around 1000 PiB ---")
test_values = [
    1000 * 2**50 - 1,  # Just below 1000 PiB
    1000 * 2**50,      # Exactly 1000 PiB
    1125899906842624000,  # Value from bug report
]

for val in test_values:
    res = format_bytes(val)
    print(f"format_bytes({val:20}) = '{res:11}' (len={len(res):2}, < 2**60: {val < 2**60})")

# Test that the assertion fails
try:
    assert len(result) <= 10
    print("\nAssertion passed (bug NOT reproduced)")
except AssertionError:
    print(f"\nAssertion failed: len('{result}') = {len(result)} > 10")
    print("Bug REPRODUCED!")

# Let's also verify 2**60 for context
print(f"\n2**60 = {2**60}")
print(f"Test value {n} is {(n / 2**60) * 100:.2f}% of 2**60")