from hypothesis import given, strategies as st
from pandas.api.indexers import FixedForwardWindowIndexer

@given(
    num_values=st.integers(min_value=1, max_value=100),
    window_size=st.integers(min_value=-20, max_value=-1),
    step=st.integers(min_value=1, max_value=10)
)
def test_fixed_forward_indexer_negative_window_size(num_values, window_size, step):
    indexer = FixedForwardWindowIndexer(window_size=window_size)
    start, end = indexer.get_window_bounds(num_values=num_values, step=step)

    for i in range(len(start)):
        assert start[i] <= end[i], f"Invalid window: start[{i}]={start[i]} > end[{i}]={end[i]}"

def test_specific_case():
    # Try with the specific failing case mentioned
    num_values = 2
    window_size = -1
    step = 1

    indexer = FixedForwardWindowIndexer(window_size=window_size)
    start, end = indexer.get_window_bounds(num_values=num_values, step=step)

    for i in range(len(start)):
        assert start[i] <= end[i], f"Invalid window: start[{i}]={start[i]} > end[{i}]={end[i]}"

# Run the test
if __name__ == "__main__":
    try:
        test_specific_case()
        print("Test with specific case passed!")
    except AssertionError as e:
        print(f"Test failed with specific case: {e}")

    # Run the property-based test
    test_fixed_forward_indexer_negative_window_size()