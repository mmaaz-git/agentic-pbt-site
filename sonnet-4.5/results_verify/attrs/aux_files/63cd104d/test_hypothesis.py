#!/usr/bin/env python3
"""Property-based test for format_bytes using Hypothesis"""

from hypothesis import given, strategies as st, settings, example
from dask.utils import format_bytes

@given(st.integers(min_value=0, max_value=2**60 - 1))
@settings(max_examples=1000, deadline=None)
@example(1125894277343089729)  # The known failing case
def test_format_bytes_length_invariant(n):
    result = format_bytes(n)
    if len(result) > 10:
        print(f"FAILED: format_bytes({n}) = {result!r} has length {len(result)} > 10")
    assert len(result) <= 10, f"format_bytes({n}) = {result!r} has length {len(result)} > 10"

if __name__ == "__main__":
    print("Running property-based test...")
    try:
        test_format_bytes_length_invariant()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")