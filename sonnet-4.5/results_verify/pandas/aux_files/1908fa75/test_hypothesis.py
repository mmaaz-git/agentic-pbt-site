#!/usr/bin/env python3
"""Hypothesis test for pandas describe() percentile formatting bug"""

from hypothesis import given, strategies as st, settings, assume
import pandas as pd

@given(
    values=st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=5, max_size=100),
    percentiles=st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False), min_size=1, max_size=10)
)
@settings(max_examples=500)
def test_describe_median_always_included(values, percentiles):
    percentiles = [p for p in percentiles if 0 <= p <= 1]
    assume(len(percentiles) > 0)
    assume(len(set(percentiles)) == len(percentiles))
    assume(0.5 not in percentiles)

    series = pd.Series(values)
    result = series.describe(percentiles=percentiles)

    assert '50%' in result.index, f"Median (50%) should always be included in describe output. Got: {result.index.tolist()}, percentiles={percentiles}"

# Run the test
if __name__ == "__main__":
    print("Running hypothesis test...")
    try:
        test_describe_median_always_included()
        print("Test passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    # Test the specific failing case
    print("\nTesting specific failing case: percentiles=[5e-324]")
    values = [1, 2, 3, 4, 5]
    percentiles = [5e-324]
    series = pd.Series(values)
    result = series.describe(percentiles=percentiles)
    print(f"Result index: {result.index.tolist()}")
    print(f"'50%' in index: {'50%' in result.index}")