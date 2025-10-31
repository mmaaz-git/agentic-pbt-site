from hypothesis import given, strategies as st, settings
import numpy as np
from pandas.core.indexers import length_of_indexer

@given(
    n=st.integers(min_value=1, max_value=1000),
    start=st.integers(min_value=-100, max_value=100) | st.none(),
    stop=st.integers(min_value=-100, max_value=100) | st.none(),
    step=st.integers(min_value=-10, max_value=10).filter(lambda x: x != 0) | st.none(),
)
@settings(max_examples=500)
def test_length_of_indexer_matches_actual(n, start, stop, step):
    target = np.arange(n)
    indexer = slice(start, stop, step)

    expected_len = length_of_indexer(indexer, target)
    actual_len = len(target[indexer])

    assert expected_len == actual_len, f"Failed: n={n}, slice({start}, {stop}, {step}) - got {expected_len}, expected {actual_len}"

if __name__ == "__main__":
    test_length_of_indexer_matches_actual()