from hypothesis import given, strategies as st, assume
from pandas.core.indexers import length_of_indexer

@given(
    start=st.integers(min_value=0, max_value=50),
    stop=st.integers(min_value=0, max_value=100),
    step=st.integers(min_value=1, max_value=10)
)
def test_length_of_indexer_range_consistency(start, stop, step):
    assume(start < stop)

    r = range(start, stop, step)
    expected_length = len(r)
    predicted_length = length_of_indexer(r)

    assert expected_length == predicted_length, \
        f"length_of_indexer({r}) returned {predicted_length} but len({r}) = {expected_length}"

if __name__ == "__main__":
    test_length_of_indexer_range_consistency()