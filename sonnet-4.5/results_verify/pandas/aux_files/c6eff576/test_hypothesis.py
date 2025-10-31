from hypothesis import given, strategies as st, assume, settings
import pyarrow as pa
from pandas.core.arrays.arrow import ArrowExtensionArray

arrow_int_array = st.lists(
    st.one_of(st.integers(min_value=-1000, max_value=1000), st.none()),
    min_size=0,
    max_size=100
).map(lambda x: ArrowExtensionArray(pa.array(x)))

@given(arrow_int_array, st.integers(min_value=0, max_value=50), st.integers(min_value=-100, max_value=100))
@settings(max_examples=200)
def test_insert_delete_inverse(arr, loc_int, item):
    assume(len(arr) > 0)
    loc = loc_int % (len(arr) + 1)

    inserted = arr.insert(loc, item)
    deleted = inserted.delete(loc)

    assert arr.equals(deleted), "delete(insert(arr, loc, item), loc) should equal arr"

# Run the test
test_insert_delete_inverse()