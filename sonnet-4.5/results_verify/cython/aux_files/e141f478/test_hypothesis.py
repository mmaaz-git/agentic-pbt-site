from hypothesis import given, strategies as st, settings
import numpy as np
from pandas.core.indexers import length_of_indexer

@given(
    start=st.one_of(st.none(), st.integers(min_value=-100, max_value=100)),
    stop=st.one_of(st.none(), st.integers(min_value=-100, max_value=100)),
    step=st.one_of(st.none(), st.integers(min_value=-100, max_value=100).filter(lambda x: x != 0)),
    n=st.integers(min_value=0, max_value=100)
)
@settings(max_examples=1000)
def test_length_of_indexer_slice_matches_actual(start, stop, step, n):
    slc = slice(start, stop, step)
    arr = np.arange(n)
    expected_len = len(arr[slc])
    computed_len = length_of_indexer(slc, arr)
    assert expected_len == computed_len, f"Failed for slice({start}, {stop}, {step}) on array of length {n}: expected {expected_len}, got {computed_len}"

if __name__ == "__main__":
    test_length_of_indexer_slice_matches_actual()
    print("Test passed!")