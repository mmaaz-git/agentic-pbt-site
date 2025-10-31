import numpy as np
from hypothesis import given, strategies as st, settings


@given(
    st.lists(
        st.tuples(
            st.text(alphabet=st.characters(min_codepoint=97, max_codepoint=122),
                    min_size=1, max_size=10),
            st.sampled_from([np.int32, np.float64])
        ),
        min_size=1,
        max_size=5,
        unique_by=lambda x: x[0]
    )
)
@settings(max_examples=200)
def test_structured_array_conversion(field_specs):
    """Test that numpy arrays with structured dtypes can be converted to ctypes objects."""
    dtype = np.dtype([(name, dt) for name, dt in field_specs])
    arr = np.zeros(10, dtype=dtype)

    # This should work but currently raises NotImplementedError
    ctypes_obj = np.ctypeslib.as_ctypes(arr)
    result = np.ctypeslib.as_array(ctypes_obj)

    # Verify round-trip conversion preserves data
    for name, _ in field_specs:
        np.testing.assert_array_equal(result[name], arr[name])


if __name__ == "__main__":
    # Run the test to find a failing case
    test_structured_array_conversion()