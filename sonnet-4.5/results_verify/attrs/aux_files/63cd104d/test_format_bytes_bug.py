#!/usr/bin/env python3
"""Test case to reproduce the format_bytes bug"""

from dask.utils import format_bytes

# Test the specific failing input
n = 1125894277343089729
result = format_bytes(n)
print(f"format_bytes({n}) = {result!r}")
print(f"Length: {len(result)}")
print(f"n < 2**60: {n < 2**60}")
print(f"2**60 = {2**60}")

# This should pass according to the documentation
try:
    assert len(result) <= 10, f"Length {len(result)} > 10"
    print("✓ Assertion passed: length <= 10")
except AssertionError as e:
    print(f"✗ Assertion failed: {e}")