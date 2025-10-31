from hypothesis import given, settings
from hypothesis.extra import numpy as npst
import numpy as np
import numpy.fft
from hypothesis import strategies as st

@given(
    npst.arrays(
        dtype=np.float64,
        shape=npst.array_shapes(min_dims=1, max_dims=3, min_side=1, max_side=20),
        elements=st.floats(min_value=-1e10, max_value=1e10, allow_nan=False, allow_infinity=False)
    )
)
@settings(max_examples=500)
def test_rfftn_irfftn_roundtrip(arr):
    result = numpy.fft.irfftn(numpy.fft.rfftn(arr))
    np.testing.assert_allclose(result, arr, rtol=1e-10, atol=1e-10)

if __name__ == "__main__":
    test_rfftn_irfftn_roundtrip()