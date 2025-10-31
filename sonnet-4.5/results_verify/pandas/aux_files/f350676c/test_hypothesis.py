from hypothesis import given, strategies as st
from pandas.core.indexers import length_of_indexer

@given(
    start=st.integers(min_value=-100, max_value=100) | st.none(),
    stop=st.integers(min_value=-100, max_value=100) | st.none(),
    step=st.integers(min_value=-10, max_value=10).filter(lambda x: x != 0) | st.none(),
    target_len=st.integers(min_value=0, max_value=100),
)
def test_length_of_indexer_matches_actual_length(start, stop, step, target_len):
    slc = slice(start, stop, step)
    target = list(range(target_len))

    expected_length = length_of_indexer(slc, target)
    actual_length = len(target[slc])

    assert expected_length == actual_length, f'Failed: slice({start}, {stop}, {step}) on target_len={target_len}. Expected {expected_length} but got {actual_length}'

# Run the test
test_length_of_indexer_matches_actual_length()