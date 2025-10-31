from hypothesis import given, strategies as st, settings
import numpy as np
from pandas.core.indexers import length_of_indexer

@given(
    target=st.lists(st.integers(), min_size=1, max_size=100),
    slice_start=st.one_of(st.none(), st.integers(min_value=-50, max_value=50)),
    slice_stop=st.one_of(st.none(), st.integers(min_value=-50, max_value=50)),
    slice_step=st.one_of(st.none(), st.integers(min_value=1, max_value=5))
)
@settings(max_examples=500)
def test_length_of_indexer_slice_positive_step_consistency(target, slice_start, slice_stop, slice_step):
    target_array = np.array(target)
    indexer = slice(slice_start, slice_stop, slice_step)

    actual_length = len(target_array[indexer])
    predicted_length = length_of_indexer(indexer, target_array)

    assert actual_length == predicted_length, (
        f"Mismatch for slice({slice_start}, {slice_stop}, {slice_step}) on array of length {len(target_array)}: "
        f"actual_length={actual_length}, predicted_length={predicted_length}"
    )

if __name__ == "__main__":
    test_length_of_indexer_slice_positive_step_consistency()