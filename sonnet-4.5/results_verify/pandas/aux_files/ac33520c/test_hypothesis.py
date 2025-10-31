from hypothesis import given, settings, strategies as st
import pandas.core.indexers as indexers

@given(
    start=st.integers(min_value=0, max_value=100),
    stop=st.integers(min_value=0, max_value=100),
    step=st.integers(min_value=1, max_value=10)
)
@settings(max_examples=500)
def test_length_of_indexer_range(start, stop, step):
    r = range(start, stop, step)
    result = indexers.length_of_indexer(r)
    expected = len(r)
    assert result == expected, f"For range({start}, {stop}, {step}): expected {expected}, got {result}"

# Run the test
test_length_of_indexer_range()