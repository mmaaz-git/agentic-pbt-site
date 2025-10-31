from hypothesis import given, strategies as st
from pandas.core.arrays.arrow import ArrowExtensionArray
import pyarrow as pa

@st.composite
def arrow_arrays(draw):
    dtype_choice = draw(st.sampled_from([pa.int64(), pa.float64(), pa.string(), pa.bool_()]))
    size = draw(st.integers(min_value=0, max_value=100))

    if dtype_choice == pa.int64():
        values = draw(st.lists(
            st.one_of(st.integers(min_value=-10000, max_value=10000), st.none()),
            min_size=size, max_size=size
        ))
    elif dtype_choice == pa.float64():
        values = draw(st.lists(
            st.one_of(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), st.none()),
            min_size=size, max_size=size
        ))
    elif dtype_choice == pa.string():
        values = draw(st.lists(st.one_of(st.text(max_size=20), st.none()), min_size=size, max_size=size))
    else:
        values = draw(st.lists(st.one_of(st.booleans(), st.none()), min_size=size, max_size=size))

    pa_array = pa.array(values, type=dtype_choice)
    return ArrowExtensionArray(pa_array)

@given(arrow_arrays())
def test_take_with_empty_indices(arr):
    result = arr.take([])
    assert len(result) == 0

if __name__ == "__main__":
    test_take_with_empty_indices()