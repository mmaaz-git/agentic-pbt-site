from hypothesis import given, strategies as st, settings, assume
import numpy as np
from scipy import integrate

@given(
    k=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    x_min=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
    x_max=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=500)
def test_tanhsinh_constant(k, x_min, x_max):
    assume(x_min < x_max)
    result = integrate.tanhsinh(lambda x: k, x_min, x_max)
    expected = k * (x_max - x_min)
    assert np.isclose(result.integral, expected, rtol=1e-10)

# Run the test
test_tanhsinh_constant()