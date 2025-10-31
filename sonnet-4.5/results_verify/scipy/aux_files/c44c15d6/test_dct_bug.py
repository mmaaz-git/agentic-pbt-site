import numpy as np
import scipy.fft
from hypothesis import given, settings
from hypothesis.extra import numpy as npst

# Property-based test from the bug report
@given(npst.arrays(
    dtype=npst.floating_dtypes(),
    shape=npst.array_shapes(min_dims=1, max_dims=1, min_side=1, max_side=100)
))
@settings(max_examples=200)
def test_dct_idct_round_trip_type1(x):
    result = scipy.fft.idct(scipy.fft.dct(x, type=1), type=1)
    assert np.allclose(result, x, rtol=1e-5, atol=1e-6)

if __name__ == "__main__":
    test_dct_idct_round_trip_type1()