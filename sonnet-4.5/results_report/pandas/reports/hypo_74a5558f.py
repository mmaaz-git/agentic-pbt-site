from hypothesis import given, strategies as st
import pytest
from pandas.api.indexers import FixedForwardWindowIndexer

@given(
    window_size=st.integers(min_value=1, max_value=100),
    num_values=st.integers(min_value=1, max_value=100),
)
def test_fixed_forward_indexer_step_zero_should_raise(window_size, num_values):
    indexer = FixedForwardWindowIndexer(window_size=window_size)

    with pytest.raises(ValueError, match="step must be"):
        indexer.get_window_bounds(num_values=num_values, step=0)

if __name__ == "__main__":
    # Run the test and capture the failure
    test_fixed_forward_indexer_step_zero_should_raise()