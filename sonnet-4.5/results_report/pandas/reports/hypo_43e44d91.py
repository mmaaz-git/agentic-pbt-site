from hypothesis import given, settings, strategies as st
import numpy as np
from pandas.core.indexers import length_of_indexer

@given(
    start=st.integers(min_value=-100, max_value=100) | st.none(),
    stop=st.integers(min_value=-100, max_value=100) | st.none(),
    step=st.integers(min_value=-100, max_value=100).filter(lambda x: x != 0) | st.none(),
    target_len=st.integers(min_value=0, max_value=200)
)
@settings(max_examples=500)
def test_length_of_indexer_slice(start, stop, step, target_len):
    slc = slice(start, stop, step)
    target = np.arange(target_len)

    computed_length = length_of_indexer(slc, target)
    actual_length = len(target[slc])

    assert computed_length == actual_length, f"Mismatch: computed={computed_length}, actual={actual_length} for slice({start}, {stop}, {step}) on array of length {target_len}"

# Run the test
if __name__ == "__main__":
    test_length_of_indexer_slice()
    print("Test passed!")  # This will only print if no assertion errors occur