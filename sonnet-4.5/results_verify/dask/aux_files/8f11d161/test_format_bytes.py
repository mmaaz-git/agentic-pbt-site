#!/usr/bin/env python3
"""Test the dask.utils.format_bytes bug report"""

from dask.utils import format_bytes

# Test the specific failing input
n = 1125894277343089729
result = format_bytes(n)

print(f"Testing specific value from bug report:")
print(f"Input: {n}")
print(f"Input < 2**60: {n < 2**60}")
print(f"Output: '{result}'")
print(f"Length: {len(result)} characters")
print(f"Expected: <= 10 characters")
print(f"Violation: {len(result) > 10}")
print()

# Test boundary conditions around 1000 PiB
print("Testing values around 1000 PiB:")
test_values = [
    999 * 2**50,  # 999 PiB
    1000 * 2**50,  # 1000 PiB
    1001 * 2**50,  # 1001 PiB
    1024 * 2**50,  # 1024 PiB
]

for val in test_values:
    output = format_bytes(val)
    print(f"  {val:20} -> '{output:15}' (length: {len(output):2}, < 2**60: {val < 2**60})")

print()

# Test various edge cases
print("Testing edge cases:")
edge_cases = [
    0,
    1,
    1023,
    1024,
    999 * 2**10,  # 999 kiB
    1000 * 2**10,  # 1000 kiB
    999 * 2**20,  # 999 MiB
    1000 * 2**20,  # 1000 MiB
    999 * 2**30,  # 999 GiB
    1000 * 2**30,  # 1000 GiB
    999 * 2**40,  # 999 TiB
    1000 * 2**40,  # 1000 TiB
    2**60 - 1,    # Maximum value that should be <= 10 chars
]

for val in edge_cases:
    output = format_bytes(val)
    length_ok = "✓" if len(output) <= 10 else "✗"
    print(f"  {val:20} -> '{output:15}' (length: {len(output):2} {length_ok})")

print()

# Run the hypothesis test
print("Running hypothesis test:")
from hypothesis import given, strategies as st, settings
import traceback

violations = []

@settings(max_examples=1000, verbosity=0)
@given(st.integers(min_value=0, max_value=2**60 - 1))
def test_format_bytes_max_length_10(n):
    result = format_bytes(n)
    if len(result) > 10:
        violations.append((n, result))

try:
    test_format_bytes_max_length_10()
    print(f"Test completed. Found {len(violations)} violations")
    if violations:
        print("First 5 violations:")
        for n, result in violations[:5]:
            print(f"  n={n:20} -> '{result}' (length: {len(result)})")
except Exception as e:
    print(f"Test failed with error: {e}")
    if violations:
        print(f"Found {len(violations)} violations before failure")
        print("First violation:")
        n, result = violations[0]
        print(f"  n={n:20} -> '{result}' (length: {len(result)})")