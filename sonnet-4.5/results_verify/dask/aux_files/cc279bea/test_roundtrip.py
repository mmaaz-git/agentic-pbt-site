#!/usr/bin/env python3
"""Test the purported bug in dask.utils format_bytes/parse_bytes round-trip."""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

from hypothesis import given, settings, strategies as st
from dask.utils import parse_bytes, format_bytes

# First, let's test the specific examples mentioned in the bug report
print("Testing specific examples from bug report:")
print("=" * 60)

test_cases = [1234, 5000, 12345, 123456]
for n in test_cases:
    formatted = format_bytes(n)
    parsed = parse_bytes(formatted)
    match = parsed == n
    print(f"Original: {n}")
    print(f"Formatted: {formatted}")
    print(f"Parsed back: {parsed}")
    print(f"Match: {match}")
    print(f"Error: {parsed - n} bytes ({((parsed - n) / n * 100):.2f}%)")
    print("-" * 40)

# Now let's run the hypothesis test
print("\nRunning Hypothesis test:")
print("=" * 60)

failures = []

@given(n=st.integers(min_value=0, max_value=2**60 - 1))
@settings(max_examples=1000, verbosity=0)
def test_format_bytes_parse_bytes_roundtrip(n):
    formatted = format_bytes(n)
    parsed = parse_bytes(formatted)
    if parsed != n:
        failures.append((n, formatted, parsed))

try:
    test_format_bytes_parse_bytes_roundtrip()
    print(f"Test completed. Found {len(failures)} failures out of 1000 tests.")
except AssertionError as e:
    print(f"AssertionError during test: {e}")

# Show some failure examples
if failures:
    print(f"\nShowing first 10 failures:")
    for n, formatted, parsed in failures[:10]:
        print(f"  {n} -> {formatted} -> {parsed} (error: {parsed - n})")

# Let's also manually verify the mathematical example from the bug report
print("\nManual verification of mathematical example:")
print("=" * 60)
n = 1234
k = 1024
ratio = n / k
print(f"1234 / 1024 = {ratio}")
print(f"Formatted as 2 decimal places: {ratio:.2f}")
print(f"{ratio:.2f} * 1024 = {float(f'{ratio:.2f}') * 1024}")
print(f"int({float(f'{ratio:.2f}') * 1024}) = {int(float(f'{ratio:.2f}') * 1024)}")

# Let's test some edge cases where round-trip DOES work
print("\nTesting cases where round-trip works:")
print("=" * 60)
working_cases = [1024, 2048, 1024*1024, 1024*1024*2]  # Exact multiples
for n in working_cases:
    formatted = format_bytes(n)
    parsed = parse_bytes(formatted)
    print(f"{n} -> {formatted} -> {parsed}, Match: {parsed == n}")