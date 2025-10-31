import numpy as np
import numpy.ctypeslib as npc
from hypothesis import given, assume, strategies as st
import hypothesis.extra.numpy as npst

@given(npst.arrays(dtype=st.just(np.dtype([('x', np.int32), ('y', np.float64)])),
                   shape=npst.array_shapes(min_dims=1, max_dims=2, max_side=10)))
def test_as_ctypes_structured_array(arr):
    """Test that as_ctypes works with structured arrays when flags allow it."""
    # Only test arrays that meet the requirements for as_ctypes
    assume(arr.flags.c_contiguous)
    assume(arr.flags.writeable)

    # This should work since as_ctypes_type works with structured dtypes
    c_arr = npc.as_ctypes(arr)

    # Verify round-trip conversion
    recovered = npc.as_array(c_arr)
    assert np.array_equal(recovered, arr)

if __name__ == "__main__":
    test_as_ctypes_structured_array()