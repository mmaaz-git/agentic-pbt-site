import numpy as np
from hypothesis import given, strategies as st, settings
from pandas.api.indexers import FixedForwardWindowIndexer

@given(
    num_values=st.integers(min_value=1, max_value=100),
    window_size=st.integers(min_value=-10, max_value=-1),
)
@settings(max_examples=10)
def test_fixed_forward_window_negative_window_size(num_values, window_size):
    indexer = FixedForwardWindowIndexer(window_size=window_size)
    start, end = indexer.get_window_bounds(num_values=num_values)

    for i in range(len(start)):
        assert start[i] <= end[i], f"At index {i}: start[{i}]={start[i]} > end[{i}]={end[i]}"

# Run the test
if __name__ == "__main__":
    try:
        test_fixed_forward_window_negative_window_size()
        print("Test passed!")
    except AssertionError as e:
        print(f"Test failed with assertion error: {e}")
    except Exception as e:
        print(f"Test failed with exception: {e}")