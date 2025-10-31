from hypothesis import given, strategies as st, assume, settings
import pyarrow as pa
from pandas.core.arrays.arrow import ArrowExtensionArray

arrow_int_array = st.lists(
    st.one_of(st.integers(min_value=-1000, max_value=1000), st.none()),
    min_size=0,
    max_size=100
).map(lambda x: ArrowExtensionArray(pa.array(x)))

@given(arrow_int_array)
@settings(max_examples=200)
def test_all_true_implies_any_true(arr):
    assume(len(arr) > 0)

    if arr.all(skipna=True):
        assert arr.any(skipna=True), "If all() is True, any() should also be True"

if __name__ == "__main__":
    # Run the test
    test_all_true_implies_any_true()
    print("Test completed")