from hypothesis import given, strategies as st
from pandas.api.indexers import FixedForwardWindowIndexer

@given(
    num_values=st.integers(min_value=1, max_value=50),
    window_size=st.integers(min_value=-10, max_value=0)
)
def test_fixed_forward_negative_window_size(num_values, window_size):
    indexer = FixedForwardWindowIndexer(window_size=window_size)
    start, end = indexer.get_window_bounds(num_values=num_values)

    for i in range(len(start)):
        assert start[i] <= end[i], f"Invariant violated at index {i}: start[{i}]={start[i]} > end[{i}]={end[i]}"

if __name__ == "__main__":
    # Run the test
    test_fixed_forward_negative_window_size()