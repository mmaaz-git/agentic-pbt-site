from hypothesis import given, strategies as st
import numpy as np
from pandas.api.indexers import FixedForwardWindowIndexer

@given(
    window_size=st.integers(min_value=-100, max_value=100),
    num_values=st.integers(min_value=1, max_value=100),
)
def test_fixed_forward_indexer_start_leq_end_invariant(window_size, num_values):
    indexer = FixedForwardWindowIndexer(window_size=window_size)
    start, end = indexer.get_window_bounds(num_values=num_values)

    assert isinstance(start, np.ndarray)
    assert isinstance(end, np.ndarray)
    assert np.all(start <= end), f"Window bounds invariant violated: start must be <= end for all indices"

if __name__ == "__main__":
    test_fixed_forward_indexer_start_leq_end_invariant()