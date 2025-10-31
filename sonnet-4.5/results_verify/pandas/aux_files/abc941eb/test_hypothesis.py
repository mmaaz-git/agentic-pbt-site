import numpy as np
from hypothesis import given, strategies as st
from pandas.api.indexers import FixedForwardWindowIndexer


@given(st.integers(min_value=1, max_value=100), st.integers(min_value=-100, max_value=-1))
def test_negative_window_size(num_values, window_size):
    indexer = FixedForwardWindowIndexer(window_size=window_size)
    start, end = indexer.get_window_bounds(num_values=num_values)
    assert np.all(start <= end), f"Found start > end with num_values={num_values}, window_size={window_size}"

# Run the test
if __name__ == "__main__":
    test_negative_window_size()