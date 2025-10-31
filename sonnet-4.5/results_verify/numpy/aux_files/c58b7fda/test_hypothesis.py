import numpy as np
import numpy.ctypeslib as npc
from hypothesis import given, assume, strategies as st
import hypothesis.extra.numpy as npst

# Test with a simple structured array first
arr = np.array([(1, 2.0)], dtype=[('x', np.int32), ('y', np.float64)])
print(f"Testing with array: {arr}")
print(f"Array dtype: {arr.dtype}")
print(f"Array flags c_contiguous: {arr.flags.c_contiguous}")
print(f"Array flags writeable: {arr.flags.writeable}")

try:
    c_arr = npc.as_ctypes(arr)
    recovered = npc.as_array(c_arr)
    print(f"Success! Recovered: {recovered}")
    print(f"Arrays equal: {np.array_equal(recovered, arr)}")
except Exception as e:
    print(f"Failed with: {type(e).__name__}: {e}")

# Now test with hypothesis
@given(npst.arrays(dtype=st.just(np.dtype([('x', np.int32), ('y', np.float64)])),
                   shape=npst.array_shapes(min_dims=1, max_dims=2, max_side=10)))
def test_as_ctypes_structured_array(arr):
    assume(arr.flags.c_contiguous)
    assume(arr.flags.writeable)

    print(f"\nHypothesis test with shape {arr.shape}")
    c_arr = npc.as_ctypes(arr)
    recovered = npc.as_array(c_arr)

    assert np.array_equal(recovered, arr)

try:
    test_as_ctypes_structured_array()
except Exception as e:
    print(f"\nHypothesis test failed: {type(e).__name__}: {e}")