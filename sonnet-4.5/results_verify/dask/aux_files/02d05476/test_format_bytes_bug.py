#!/usr/bin/env python3
"""Test the format_bytes bug report"""

from hypothesis import given, settings, strategies as st
from dask.utils import format_bytes

# First, run the hypothesis test
print("Running Hypothesis test...")
@given(st.integers(min_value=0, max_value=2**60 - 1))
@settings(max_examples=1000)
def test_format_bytes_length_claim(n):
    """
    Test the documented claim: "For all values < 2**60, the output is always <= 10 characters."
    """
    result = format_bytes(n)
    assert len(result) <= 10, f"format_bytes({n}) = '{result}' has length {len(result)}, expected <= 10"

try:
    test_format_bytes_length_claim()
    print("Hypothesis test passed")
except AssertionError as e:
    print(f"Hypothesis test failed: {e}")

# Now run the specific reproduction case
print("\nRunning specific reproduction case...")
result = format_bytes(1_125_899_906_842_624_000)
print(f"Result: '{result}'")
print(f"Length: {len(result)} characters")

try:
    assert len(result) <= 10, f"Expected <= 10 characters, got {len(result)}"
    print("Assertion passed")
except AssertionError as e:
    print(f"Assertion failed: {e}")

# Test additional examples
print("\nAdditional examples:")
examples = [
    (999 * 2**50, "999 * 2**50"),
    (1000 * 2**50, "1000 * 2**50"),
    (1023 * 2**50, "1023 * 2**50"),
]

for value, desc in examples:
    result = format_bytes(value)
    print(f"format_bytes({desc}) = '{result}' (length: {len(result)})")