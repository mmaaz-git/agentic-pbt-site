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
    if len(result) > 10:
        print(f"FAILURE: format_bytes({n}) = '{result}' has length {len(result)} > 10")
        return False
    return True

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