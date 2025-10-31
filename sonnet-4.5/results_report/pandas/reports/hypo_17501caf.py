import numpy as np
from hypothesis import given, strategies as st, settings
from pandas.api.indexers import FixedForwardWindowIndexer


@settings(max_examples=1000)
@given(
    num_values=st.integers(min_value=1, max_value=100),
    window_size=st.integers(min_value=-100, max_value=100),
    step=st.integers(min_value=1, max_value=10)
)
def test_fixed_forward_window_start_le_end_invariant(num_values, window_size, step):
    indexer = FixedForwardWindowIndexer(window_size=window_size)
    start, end = indexer.get_window_bounds(num_values=num_values, step=step)

    for i in range(len(start)):
        assert start[i] <= end[i], f"Invalid window bounds at index {i}: start={start[i]}, end={end[i]}"


if __name__ == "__main__":
    # Run the test
    test_fixed_forward_window_start_le_end_invariant()