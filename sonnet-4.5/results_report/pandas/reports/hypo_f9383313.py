from hypothesis import given, strategies as st
from pandas.core.indexers import length_of_indexer

@given(
    start=st.integers(min_value=0, max_value=100),
    stop=st.integers(min_value=0, max_value=100),
    step=st.integers(min_value=1, max_value=10)
)
def test_length_of_indexer_range_consistency(start, stop, step):
    rng = range(start, stop, step)
    expected_length = len(rng)
    predicted_length = length_of_indexer(rng)

    assert expected_length == predicted_length, \
        f"For range({start}, {stop}, {step}): expected {expected_length}, got {predicted_length}"

if __name__ == "__main__":
    test_length_of_indexer_range_consistency()