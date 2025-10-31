from hypothesis import given, settings, strategies as st
from pandas.core.indexers import length_of_indexer

@given(
    start=st.integers(min_value=0, max_value=20),
    stop=st.integers(min_value=0, max_value=20),
    step=st.integers(min_value=1, max_value=5)
)
@settings(max_examples=300)
def test_length_of_indexer_range(start, stop, step):
    indexer = range(start, stop, step)

    computed = length_of_indexer(indexer)
    expected = len(list(indexer))

    assert computed == expected, f"Failed for range({start}, {stop}, {step}): computed={computed}, expected={expected}"

# Run the test
print("Running property-based test...")
try:
    test_length_of_indexer_range()
    print("All tests passed!")
except AssertionError as e:
    print(f"Test failed: {e}")