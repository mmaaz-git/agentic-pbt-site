from hypothesis import given, assume, settings
import hypothesis.strategies as st
from pandas.core.indexers.utils import length_of_indexer

@given(
    st.integers(min_value=0, max_value=1000),
    st.integers(min_value=0, max_value=100),
    st.integers(min_value=1, max_value=20)
)
@settings(max_examples=300)
def test_range_length_of_indexer_property(start, stop, step):
    assume(start < stop)
    indexer = range(start, stop, step)

    computed_length = length_of_indexer(indexer)
    expected_length = len(indexer)

    if computed_length != expected_length:
        print(f"FAILURE: range({start}, {stop}, {step})")
        print(f"  computed: {computed_length}, expected: {expected_length}")
        assert False, f"Mismatch for range({start}, {stop}, {step})"

print("Running property-based test...")
test_range_length_of_indexer_property()