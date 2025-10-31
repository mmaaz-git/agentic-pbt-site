from hypothesis import given, strategies as st
import numpy as np
from pandas.api.indexers import FixedForwardWindowIndexer


@given(st.integers(min_value=1, max_value=100), st.integers(min_value=-10, max_value=-1))
def test_fixed_forward_window_negative_produces_invalid_bounds(num_values, window_size):
    indexer = FixedForwardWindowIndexer(window_size=window_size)
    start, end = indexer.get_window_bounds(num_values=num_values)

    for i in range(len(start)):
        assert start[i] <= end[i], f"Window bounds violated: start[{i}]={start[i]} > end[{i}]={end[i]}"

if __name__ == "__main__":
    # Run with the specific failing example
    try:
        indexer = FixedForwardWindowIndexer(window_size=-1)
        start, end = indexer.get_window_bounds(num_values=2)

        for i in range(len(start)):
            assert start[i] <= end[i], f"Window bounds violated: start[{i}]={start[i]} > end[{i}]={end[i]}"
        print("Test passed with specific example")
    except AssertionError as e:
        print(f"Assertion failed as expected: {e}")