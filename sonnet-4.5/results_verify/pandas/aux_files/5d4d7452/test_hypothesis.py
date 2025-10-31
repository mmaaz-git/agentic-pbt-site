from hypothesis import given, strategies as st, settings
from pandas.core.indexers.utils import length_of_indexer

@given(
    st.integers(min_value=-20, max_value=20),
    st.integers(min_value=-20, max_value=20),
    st.one_of(st.integers(min_value=-5, max_value=5).filter(lambda x: x != 0), st.none())
)
@settings(max_examples=100)  # Reduced for faster testing
def test_length_of_indexer_slice_property(start, stop, step):
    target = list(range(50))
    slc = slice(start, stop, step)

    computed_length = length_of_indexer(slc, target)
    actual_length = len(target[slc])

    if computed_length != actual_length:
        print(f"Failed on slice({start}, {stop}, {step})")
        print(f"  Computed: {computed_length}, Actual: {actual_length}")
        assert False, f"Mismatch for slice({start}, {stop}, {step})"

# Run the test
print("Running property-based test...")
try:
    test_length_of_indexer_slice_property()
    print("Test passed!")
except Exception as e:
    print(f"Test failed: {e}")