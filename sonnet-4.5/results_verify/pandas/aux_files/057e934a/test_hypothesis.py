import pytest
from hypothesis import given, strategies as st, settings, Verbosity
from pandas.api.indexers import FixedForwardWindowIndexer


@given(st.integers(min_value=1, max_value=100), st.integers(max_value=-1))
@settings(verbosity=Verbosity.verbose, max_examples=10)
def test_fixed_forward_window_negative_size_start_end_invariant(num_values, window_size):
    print(f"Testing with num_values={num_values}, window_size={window_size}")
    indexer = FixedForwardWindowIndexer(window_size=window_size)
    start, end = indexer.get_window_bounds(num_values)

    for i in range(len(start)):
        assert start[i] <= end[i]

if __name__ == "__main__":
    # Run the test
    test_fixed_forward_window_negative_size_start_end_invariant()