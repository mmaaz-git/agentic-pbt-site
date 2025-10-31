from hypothesis import given, assume, settings, HealthCheck
import hypothesis.strategies as st
from pandas.core.indexers.utils import length_of_indexer

@given(
    st.integers(min_value=0, max_value=100),
    st.integers(min_value=1, max_value=100),
    st.integers(min_value=1, max_value=20)
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.filter_too_much])
def test_range_length_of_indexer_property(start, length, step):
    stop = start + length  # Generate stop based on start and desired length
    indexer = range(start, stop, step)

    computed_length = length_of_indexer(indexer)
    expected_length = len(indexer)

    if computed_length != expected_length:
        print(f"FAILURE: range({start}, {stop}, {step})")
        print(f"  computed: {computed_length}, expected: {expected_length}")
        assert False, f"Mismatch for range({start}, {stop}, {step})"

print("Running property-based test...")
try:
    test_range_length_of_indexer_property()
    print("Property test completed without finding errors (but we know range(0,1,2) fails)")
except AssertionError as e:
    print(f"Property test found failure: {e}")