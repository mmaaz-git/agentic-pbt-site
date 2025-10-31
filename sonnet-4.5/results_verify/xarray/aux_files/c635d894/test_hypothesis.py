import numpy as np
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from xarray.coding.variables import CFScaleOffsetCoder
from xarray.core.variable import Variable
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

@given(arrays(dtype=np.float32, shape=st.tuples(st.integers(5, 20))))
@settings(max_examples=50)
def test_scale_offset_coder_zero_scale(data):
    assume(not np.any(np.isnan(data)))
    assume(not np.any(np.isinf(data)))

    scale_factor = 0.0
    add_offset = 10.0

    original_var = Variable(('x',), data.copy(),
                          encoding={'scale_factor': scale_factor, 'add_offset': add_offset})
    coder = CFScaleOffsetCoder()

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        encoded_var = coder.encode(original_var)

    decoded_var = coder.decode(encoded_var)

    try:
        np.testing.assert_array_equal(original_var.data, decoded_var.data)
        print(f"✓ Test passed with data: {data[:5]}...")
    except AssertionError:
        print(f"✗ Test FAILED with data: {data[:5]}...")
        print(f"  Original: {original_var.data[:5]}...")
        print(f"  Decoded:  {decoded_var.data[:5]}...")
        return False
    return True

# Run the test
print("Running hypothesis test with scale_factor=0.0...")
print("=" * 50)

try:
    test_scale_offset_coder_zero_scale()
except Exception as e:
    print(f"Test failed with exception: {e}")