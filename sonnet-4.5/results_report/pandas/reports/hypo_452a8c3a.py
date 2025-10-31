from hypothesis import given, strategies as st, assume
import numpy as np
from scipy import special

@given(
    st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False)
)
def test_boxcox_inv_boxcox_roundtrip(lmbda, x):
    y = special.boxcox(x, lmbda)
    assume(not np.isinf(y) and not np.isnan(y))
    x_recovered = special.inv_boxcox(y, lmbda)
    assert np.isclose(x, x_recovered, rtol=1e-6, atol=1e-6), f"Failed for lmbda={lmbda}, x={x}: recovered {x_recovered}"

if __name__ == "__main__":
    test_boxcox_inv_boxcox_roundtrip()