import numpy as np
import numpy.ma as ma
from hypothesis import given, strategies as st, settings

@given(st.sampled_from([np.int32, np.int64, np.float32, np.float64]))
@settings(max_examples=100)
def test_fill_value_functions_accept_dtype_classes(dtype):
    default = ma.default_fill_value(dtype)
    maximum = ma.maximum_fill_value(dtype)
    minimum = ma.minimum_fill_value(dtype)
    assert all(x is not None for x in [default, maximum, minimum])

# Run the test
test_fill_value_functions_accept_dtype_classes()