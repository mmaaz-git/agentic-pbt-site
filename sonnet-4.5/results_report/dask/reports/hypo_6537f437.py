#!/usr/bin/env python3
from hypothesis import given, settings, strategies as st
from dask.utils import format_bytes

@given(st.integers(min_value=0, max_value=2**60 - 1))
@settings(max_examples=1000)
def test_format_bytes_length_claim(n):
    """
    Test the documented claim: "For all values < 2**60, the output is always <= 10 characters."
    """
    result = format_bytes(n)
    assert len(result) <= 10, f"format_bytes({n}) = '{result}' has length {len(result)}, expected <= 10"

if __name__ == "__main__":
    test_format_bytes_length_claim()