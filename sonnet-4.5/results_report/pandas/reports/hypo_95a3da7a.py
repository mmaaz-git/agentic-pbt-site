from hypothesis import given, strategies as st, settings
from pandas.core.indexers.objects import FixedWindowIndexer
import numpy as np

@settings(max_examples=1000)
@given(
    num_values=st.integers(min_value=0, max_value=200),
    window_size=st.integers(min_value=0, max_value=50),
    center=st.booleans(),
    closed=st.sampled_from([None, "left", "right", "both", "neither"]),
    step=st.integers(min_value=1, max_value=10) | st.none(),
)
def test_fixed_window_indexer_comprehensive(num_values, window_size, center, closed, step):
    indexer = FixedWindowIndexer(window_size=window_size)
    start, end = indexer.get_window_bounds(
        num_values=num_values,
        center=center,
        closed=closed,
        step=step
    )

    assert np.all(start <= end), f"start <= end should hold for all windows. Got start={start}, end={end}"

if __name__ == "__main__":
    test_fixed_window_indexer_comprehensive()