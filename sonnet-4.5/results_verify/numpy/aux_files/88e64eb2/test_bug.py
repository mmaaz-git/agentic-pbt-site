import numpy as np
import numpy.ma as ma
from hypothesis import given, strategies as st, settings
from hypothesis.extra import numpy as npst

@st.composite
def float_masked_arrays(draw):
    shape = draw(npst.array_shapes(max_dims=1, max_side=20))
    data = draw(npst.arrays(dtype=np.float64, shape=shape,
                           elements=st.floats(allow_nan=False, allow_infinity=False,
                                            min_value=-100, max_value=100)))
    mask = draw(npst.arrays(dtype=bool, shape=shape))
    return ma.array(data, mask=mask)

@given(float_masked_arrays())
@settings(max_examples=500)
def test_make_mask_descr_consistency(arr):
    mask = ma.make_mask(ma.getmaskarray(arr))
    assert mask.shape == arr.shape

if __name__ == "__main__":
    test_make_mask_descr_consistency()