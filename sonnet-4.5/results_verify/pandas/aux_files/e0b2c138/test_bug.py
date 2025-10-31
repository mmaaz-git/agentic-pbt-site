import numpy as np
from hypothesis import given, strategies as st, settings
from pandas.core.indexers import length_of_indexer

@given(
    target_len=st.integers(min_value=5, max_value=50),
    start=st.integers(min_value=0, max_value=49),
    stop=st.integers(min_value=0, max_value=49),
)
@settings(max_examples=1000)
def test_length_of_indexer_never_negative(target_len, start, stop):
    target = np.arange(target_len)
    indexer = slice(start, stop, 1)

    actual_length = len(target[indexer])
    predicted_length = length_of_indexer(indexer, target)

    assert predicted_length >= 0, f"length_of_indexer returned negative: {predicted_length}"
    assert predicted_length == actual_length

if __name__ == "__main__":
    test_length_of_indexer_never_negative()
    print("Test completed")