import numpy as np
from hypothesis import given, strategies as st, example
from pandas.core.indexers import length_of_indexer


@given(
    st.integers(min_value=1, max_value=100),
    st.integers(min_value=-100, max_value=100).filter(lambda x: x != 0),
    st.integers(min_value=-100, max_value=100),
    st.integers(min_value=-100, max_value=100),
)
@example(1, 1, 1, 0)  # Add the specific failing example
def test_length_of_indexer_slice_matches_actual_length(target_len, step, start, stop):
    target = list(range(target_len))
    slc = slice(start, stop, step)

    calculated_length = length_of_indexer(slc, target)
    actual_slice = target[slc]
    actual_length = len(actual_slice)

    assert calculated_length == actual_length, \
        f"length_of_indexer({slc}, target={target_len}) = {calculated_length}, but len(target[{slc}]) = {actual_length}"


if __name__ == "__main__":
    # Run the test
    test_length_of_indexer_slice_matches_actual_length()