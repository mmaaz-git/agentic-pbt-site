import numpy as np
from pandas.core.indexers import length_of_indexer
from hypothesis import given, strategies as st, settings

@given(
    start=st.integers(min_value=0, max_value=100),
    stop=st.integers(min_value=0, max_value=100),
    step=st.integers(min_value=1, max_value=10),
    target_len=st.integers(min_value=1, max_value=100)
)
@settings(max_examples=1000)
def test_length_of_indexer_matches_actual_length(start, stop, step, target_len):
    slc = slice(start, stop, step)
    target = np.arange(target_len)

    computed_length = length_of_indexer(slc, target)
    actual_length = len(target[slc])

    assert computed_length == actual_length, f"Mismatch for slice({start}, {stop}, {step}) on array of length {target_len}: computed={computed_length}, actual={actual_length}"

if __name__ == "__main__":
    # Run the test
    test_length_of_indexer_matches_actual_length()