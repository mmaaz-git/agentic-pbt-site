from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays
import numpy as np
import scipy.ndimage as ndi

@given(
    value=st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False),
    rows=st.integers(5, 15),
    cols=st.integers(5, 15)
)
@settings(max_examples=50, deadline=None)
def test_sobel_constant_zero(value, rows, cols):
    """
    Property: sobel(constant_image) = 0

    Sobel filter detects edges. A constant image has no edges,
    so the result should be zero everywhere.
    """
    x = np.full((rows, cols), value, dtype=np.float64)
    result = ndi.sobel(x, mode='constant', cval=value)

    assert np.allclose(result, 0.0, atol=1e-10), \
        f"Sobel on constant not zero: max = {np.max(np.abs(result))}"

# Run the test
test_sobel_constant_zero()