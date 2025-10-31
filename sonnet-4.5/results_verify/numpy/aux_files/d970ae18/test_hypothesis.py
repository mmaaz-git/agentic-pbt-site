import ctypes
import numpy as np
from hypothesis import assume, given, settings
from hypothesis.extra import numpy as npst

@given(npst.arrays(dtype=npst.scalar_dtypes(), shape=npst.array_shapes(min_dims=1, max_dims=3)))
@settings(max_examples=100)
def test_as_array_from_pointer_with_shape(arr):
    assume(arr.flags.c_contiguous)
    assume(arr.dtype.hasobject == False)

    try:
        ct_arr = np.ctypeslib.as_ctypes(arr)
    except (TypeError, NotImplementedError):
        assume(False)

    ptr = ctypes.cast(ct_arr, ctypes.POINTER(ct_arr._type_))
    result = np.ctypeslib.as_array(ptr, arr.shape)

    np.testing.assert_array_equal(result, arr)
    assert result.shape == arr.shape, f"Expected shape {arr.shape}, got {result.shape}"

# Run the test
if __name__ == "__main__":
    test_as_array_from_pointer_with_shape()
    print("Test completed")