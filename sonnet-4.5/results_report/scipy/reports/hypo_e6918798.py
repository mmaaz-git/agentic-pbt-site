from hypothesis import given, strategies as st, assume, settings
import scipy.special
import numpy as np
import math

@given(
    x=st.floats(min_value=-0.99, max_value=1e6, allow_nan=False, allow_infinity=False),
    lmbda=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=1000)
def test_boxcox1p_inv_boxcox1p_roundtrip(x, lmbda):
    y = scipy.special.boxcox1p(x, lmbda)
    assume(not np.isnan(y) and not np.isinf(y))
    result = scipy.special.inv_boxcox1p(y, lmbda)
    assert math.isclose(result, x, rel_tol=1e-9, abs_tol=1e-12)

# Run the test
if __name__ == "__main__":
    test_boxcox1p_inv_boxcox1p_roundtrip()