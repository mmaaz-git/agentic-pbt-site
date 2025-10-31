import numpy as np
import numpy.ctypeslib
from hypothesis import given, settings
import hypothesis.extra.numpy as npst


@settings(max_examples=300)
@given(npst.arrays(dtype=npst.scalar_dtypes(), shape=npst.array_shapes(min_dims=2)))
def test_as_ctypes_supports_fortran_order(arr):
    f_arr = np.asfortranarray(arr)
    ct = np.ctypeslib.as_ctypes(f_arr)
    print(f"Test passed for array shape {arr.shape} dtype {arr.dtype}")

if __name__ == "__main__":
    test_as_ctypes_supports_fortran_order()