from hypothesis import given, strategies as st
import pandas.arrays as arrays
import numpy as np

@given(st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=50))
def test_numpy_extension_array_wrapper_consistency(values):
    np_arr = np.array(values)
    numpy_ext_arr = arrays.NumpyExtensionArray(np_arr)

    # This assertion fails!
    assert numpy_ext_arr.dtype == np_arr.dtype, \
        f"Dtype mismatch: {numpy_ext_arr.dtype} != {np_arr.dtype}"

# Run the test
test_numpy_extension_array_wrapper_consistency()