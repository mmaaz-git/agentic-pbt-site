import numpy as np
import numpy.ma as ma
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra import numpy as npst

@st.composite
def masked_2d_arrays(draw):
    shape = draw(npst.array_shapes(min_dims=2, max_dims=2, max_side=10))
    data = draw(npst.arrays(dtype=np.int64, shape=shape,
                           elements=st.integers(min_value=-100, max_value=100)))
    mask = draw(npst.arrays(dtype=bool, shape=shape))
    return ma.array(data, mask=mask)

@given(masked_2d_arrays())
@settings(max_examples=1000)
def test_compress_rowcols_maintains_2d(arr):
    result = ma.compress_rowcols(arr)
    assert result.ndim == 2

if __name__ == "__main__":
    test_compress_rowcols_maintains_2d()