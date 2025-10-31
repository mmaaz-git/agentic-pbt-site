#!/usr/bin/env python3
"""Run the hypothesis test from the bug report"""

from hypothesis import given, strategies as st, settings
from dask.utils import format_bytes

@given(st.integers(min_value=0, max_value=2**60 - 1))
@settings(max_examples=1000)
def test_format_bytes_length_property(n):
    """For all values < 2**60, output should be <= 10 characters."""
    result = format_bytes(n)
    assert len(result) <= 10, f"Output exceeds 10 chars for {n}: '{result}' (len={len(result)})"

# Run the test
if __name__ == "__main__":
    try:
        test_format_bytes_length_property()
        print("Test passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")