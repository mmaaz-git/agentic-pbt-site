from hypothesis import given, strategies as st, settings
import numpy as np
from pandas.core.indexers import length_of_indexer

@given(
    start=st.none() | st.integers(min_value=-50, max_value=50),
    stop=st.none() | st.integers(min_value=-50, max_value=50),
    step=st.none() | st.integers(min_value=-10, max_value=10).filter(lambda x: x != 0),
)
@settings(max_examples=500)
def test_length_of_indexer_matches_actual_slice(start, stop, step):
    target_len = 50
    target = np.arange(target_len)
    slc = slice(start, stop, step)
    expected_len = length_of_indexer(slc, target)
    actual_sliced = target[slc]
    actual_len = len(actual_sliced)
    print(f"Testing slice({start}, {stop}, {step}): expected={expected_len}, actual={actual_len}")
    assert expected_len == actual_len, f"Mismatch for slice({start}, {stop}, {step}): expected={expected_len}, actual={actual_len}"

if __name__ == "__main__":
    # Run the test
    test_length_of_indexer_matches_actual_slice()