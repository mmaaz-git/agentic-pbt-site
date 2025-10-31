import numpy as np
import numpy.ma as ma
from hypothesis import given, strategies as st, settings, example
from hypothesis.extra import numpy as npst

@st.composite
def masked_arrays(draw, dtype=np.int64, max_dims=1, max_side=20):
    shape = draw(npst.array_shapes(max_dims=max_dims, max_side=max_side))
    data = draw(npst.arrays(dtype=dtype, shape=shape))
    mask = draw(npst.arrays(dtype=bool, shape=shape))
    return ma.array(data, mask=mask)

@given(masked_arrays())
@settings(max_examples=500)
@example(ma.array([999, 9223372036854775807, 888], mask=[True, False, True]))
def test_unique_treats_all_masked_as_one(arr):
    unique_vals = ma.unique(arr)
    masked_count = sum(1 for val in unique_vals if ma.is_masked(val))
    if ma.getmaskarray(arr).any():
        try:
            assert masked_count <= 1, f"Expected at most 1 masked value, got {masked_count}. Input: {arr}, Unique: {unique_vals}"
        except AssertionError as e:
            print(f"FAILURE: {e}")
            return False
    return True

# Run the test
print("Running hypothesis test...")
try:
    test_unique_treats_all_masked_as_one()
    print("All tests passed!")
except Exception as e:
    print(f"Test failed: {e}")
