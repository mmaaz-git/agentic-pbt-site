#!/usr/bin/env python3
"""Run the hypothesis test from the bug report"""

from hypothesis import given, strategies as st, settings
import dask.dataframe.dask_expr as de

@given(st.floats(min_value=0, max_value=1e20, allow_nan=False, allow_infinity=False))
@settings(max_examples=100)
def test_memory_repr_returns_string_with_unit(num):
    result = de.memory_repr(num)
    assert isinstance(result, str), f"Expected string, got {type(result).__name__} for input {num}"

# Run the test
if __name__ == "__main__":
    try:
        test_memory_repr_returns_string_with_unit()
        print("✓ All hypothesis tests passed")
    except AssertionError as e:
        print(f"✗ Hypothesis test failed: {e}")