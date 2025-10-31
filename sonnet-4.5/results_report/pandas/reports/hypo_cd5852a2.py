from hypothesis import given, strategies as st
import numpy as np
from pandas.core.indexers import length_of_indexer


@given(
    start=st.integers(min_value=0, max_value=100) | st.none(),
    stop=st.integers(min_value=0, max_value=100) | st.none(),
    step=st.integers(min_value=1, max_value=10) | st.none(),
    target_len=st.integers(min_value=0, max_value=100),
)
def test_length_of_indexer_slice_matches_actual(start, stop, step, target_len):
    slc = slice(start, stop, step)
    target = np.arange(target_len)

    computed_length = length_of_indexer(slc, target)
    actual_length = len(target[slc])

    assert computed_length == actual_length, \
        f"Mismatch for slice({start}, {stop}, {step}) on target of length {target_len}: " \
        f"computed={computed_length}, actual={actual_length}"


if __name__ == "__main__":
    # Run the test
    test_length_of_indexer_slice_matches_actual()