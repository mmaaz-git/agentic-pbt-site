#!/usr/bin/env python3
"""Property-based test for format_bytes using Hypothesis"""

from hypothesis import given, strategies as st, settings, example
from dask.utils import format_bytes

@given(st.integers(min_value=0, max_value=2**60 - 1))
@settings(max_examples=1000)
@example(1125894277343089729)  # Value from bug report
@example(1125899906842624000)  # Another problematic value
def test_format_bytes_length_invariant(n):
    result = format_bytes(n)
    assert len(result) <= 10, f"format_bytes({n}) = '{result}' has length {len(result)} > 10"

# Run the test
print("Running Hypothesis property-based test...")
try:
    test_format_bytes_length_invariant()
    print("All tests passed!")
except AssertionError as e:
    print(f"Test failed: {e}")

# Let's also find the exact threshold where it breaks
print("\n--- Finding the exact threshold ---")
for multiplier in [999, 999.9, 999.99, 1000]:
    val = int(multiplier * 2**50)
    res = format_bytes(val)
    print(f"{multiplier:7.2f} * 2^50 = {val:20} -> '{res:11}' (len={len(res)})")

# Verify the math on when this breaks
print("\n--- Understanding when the bug occurs ---")
print("The function uses n / k where k = 2**50 for PiB")
print("It formats with .2f, which means 2 decimal places")
print("When n / 2**50 >= 1000, we get 4 or more digits before the decimal")
print(f"1000 * 2**50 = {1000 * 2**50}")
print(f"This is still < 2**60 = {2**60}")
print(f"In fact, 1000 * 2**50 / 2**60 = {(1000 * 2**50) / 2**60:.2%} of 2**60")