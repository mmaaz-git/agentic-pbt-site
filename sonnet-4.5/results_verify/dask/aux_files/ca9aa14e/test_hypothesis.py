#!/usr/bin/env python3
"""Hypothesis test for format_bytes bug"""

from hypothesis import given, strategies as st, settings, example
from dask.utils import format_bytes

@given(st.integers(min_value=1, max_value=2**60 - 1))
@example(1_125_894_277_343_089_729)  # The failing example from the report
@example(1_125_899_906_842_624_000)  # 1000 * 2^50
@settings(max_examples=1000)
def test_format_bytes_output_length(n):
    result = format_bytes(n)
    if len(result) > 10:
        print(f"Found violation: n={n}, result={result!r}, len={len(result)}")
    assert len(result) <= 10, f"Output too long for n={n}: {result!r} has {len(result)} characters"

if __name__ == "__main__":
    print("Running Hypothesis test...")
    try:
        test_format_bytes_output_length()
        print("Test passed with all examples!")
    except AssertionError as e:
        print(f"Test failed: {e}")