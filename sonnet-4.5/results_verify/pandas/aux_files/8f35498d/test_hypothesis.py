from hypothesis import given, strategies as st, settings
import numpy as np
from pandas.core.indexers import length_of_indexer

@given(
    target=st.lists(st.integers(), min_size=1, max_size=50),
    slice_start=st.one_of(st.none(), st.integers(min_value=-50, max_value=50)),
    slice_stop=st.one_of(st.none(), st.integers(min_value=-50, max_value=50)),
    slice_step=st.one_of(st.none(), st.integers(min_value=1, max_value=10))
)
@settings(max_examples=100)
def test_length_of_indexer_slice_consistency(target, slice_start, slice_stop, slice_step):
    target_array = np.array(target)
    indexer = slice(slice_start, slice_stop, slice_step)

    actual_length = len(target_array[indexer])
    predicted_length = length_of_indexer(indexer, target_array)

    print(f"Testing: target_len={len(target)}, indexer={indexer}")
    print(f"  Actual: {actual_length}, Predicted: {predicted_length}")

    assert actual_length == predicted_length, f"Mismatch for target_len={len(target)}, indexer={indexer}: actual={actual_length}, predicted={predicted_length}"

if __name__ == "__main__":
    test_length_of_indexer_slice_consistency()