import numpy as np
from hypothesis import given, settings, strategies as st
from pandas.api.indexers import FixedForwardWindowIndexer

@given(
    num_values=st.integers(min_value=1, max_value=100),
    window_size=st.integers(min_value=-100, max_value=-1),
)
@settings(max_examples=500)
def test_fixed_forward_negative_window_size(num_values, window_size):
    indexer = FixedForwardWindowIndexer(window_size=window_size)
    start, end = indexer.get_window_bounds(num_values=num_values)

    for i in range(len(start)):
        assert start[i] <= end[i], f"start[{i}]={start[i]} > end[{i}]={end[i]}"

if __name__ == "__main__":
    test_fixed_forward_negative_window_size()